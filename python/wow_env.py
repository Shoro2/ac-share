import gymnasium as gym
from gymnasium import spaces
import numpy as np
import socket
import json
import time
import math
import os

SPELL_SMITE = 585
SPELL_HEAL = 2050
MEMORY_FILE = "npc_memory.json"

class WoWEnv(gym.Env):
    def __init__(self, host='127.0.0.1', port=5000, bot_name=None):
        super(WoWEnv, self).__init__()
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((host, port))
            self.sock.setblocking(True)
            self.sock.settimeout(2.0)
            print(">>> WoW Env v11 (Aggro & Gear) verbunden! <<<")
        except Exception as e:
            print(f"VERBINDUNGSFEHLER: {e}")
            raise e

        # Action Space: 8=SELL
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=-1.0, high=float('inf'), shape=(10,), dtype=np.float32)

        self.last_state = None
        self.my_name = ""
        self.bot_name = bot_name
        self._recv_buffer = ""
        
        self.npc_memory = {} 
        self.blacklist = {} 
        self.BLACKLIST_DURATION = 15 * 60 
        
        self.load_memory()
        self.last_save_time = time.time()

    def load_memory(self):
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, 'r') as f:
                    self.npc_memory = json.load(f)
            except: self.npc_memory = {}

    def save_memory(self):
        try:
            with open(MEMORY_FILE, 'w') as f:
                json.dump(self.npc_memory, f, indent=4)
        except: pass

    def _get_state_from_server(self, allow_any_player=False):
        while True:
            try:
                data = self.sock.recv(8192) 
                if not data: break
                self._recv_buffer += data.decode('utf-8')
                if "\n" in self._recv_buffer:
                    lines = self._recv_buffer.split("\n")
                    if len(lines) < 2: continue
                    raw_json = lines[-2]
                    self._recv_buffer = lines[-1]
                    if not raw_json.strip(): continue
                    try:
                        data = json.loads(raw_json)
                        if not data['players']: continue
                        player = None
                        if self.bot_name:
                            for candidate in data['players']:
                                if candidate.get('name') == self.bot_name:
                                    player = candidate
                                    break
                        if player is None:
                            if self.bot_name and not allow_any_player:
                                return None
                            player = data['players'][0]
                        self.my_name = player['name']
                        return player
                    except: continue
            except socket.timeout:
                return None
            except Exception: continue
        return None

    def _manage_blacklist(self):
        now = time.time()
        self.blacklist = {guid: expiry for guid, expiry in self.blacklist.items() if expiry > now}

    def get_best_memory_target(self):
        if not self.last_state: return None
        me_pos = {'x': self.last_state['x'], 'y': self.last_state['y'], 'z': self.last_state['z']}
        best_mob = None
        min_dist = 9999.0
        
        for guid, mob in self.npc_memory.items():
            if guid in self.blacklist: continue
            is_dead = (mob['hp'] == 0)
            dx = mob['x'] - me_pos['x']
            dy = mob['y'] - me_pos['y']
            dist = math.sqrt(dx*dx + dy*dy)
            
            if is_dead and dist > 40.0: continue
            if dist > 5.0 and dist < min_dist:
                min_dist = dist
                best_mob = mob
        return best_mob

    def step(self, action):
        self._manage_blacklist()
        if time.time() - self.last_save_time > 30.0:
            self.save_memory()
            self.last_save_time = time.time()
        
        override_action = action
        
        nearby_mobs = []
        my_lvl = 1
        hp_pct = 1.0 
        dist_to_target = 9999.0
        free_slots = 0
        
        if self.last_state:
            nearby_mobs = self.last_state.get('nearby_mobs', [])
            my_lvl = self.last_state.get('level', 1)
            max_hp = max(1, self.last_state['max_hp'])
            hp_pct = self.last_state['hp'] / max_hp
            free_slots = self.last_state.get('free_slots', 20)
            
            if self.last_state.get('tx') != 0:
                dx = self.last_state['tx'] - self.last_state['x']
                dy = self.last_state['ty'] - self.last_state['y']
                dist_to_target = math.sqrt(dx*dx + dy*dy)
            
            in_combat = (self.last_state['combat'] == 'true')
            target_alive = (self.last_state['target_status'] == 'alive')
            target_dead = (self.last_state['target_status'] == 'dead')
            is_casting = (self.last_state.get('casting') == 'true')
            
            # --- OVERRIDE LOGIK ---
            
            # 1. Vendor Check (Nur bei wirklich voller Tasche)
            vendor_mode = False
            needs_vendor = (free_slots < 2) 
            
            best_vendor = None
            min_v_dist = 9999.0
            me_pos = {'x': self.last_state['x'], 'y': self.last_state['y'], 'z': self.last_state['z']}
            
            if needs_vendor:
                for guid, mob in self.npc_memory.items():
                    if mob.get('vendor', 0) == 1:
                        dx = mob['x'] - me_pos['x']
                        dy = mob['y'] - me_pos['y']
                        d = math.sqrt(dx*dx + dy*dy)
                        if d < min_v_dist:
                            min_v_dist = d
                            best_vendor = mob

            # Prio 1: Verkaufen
            if needs_vendor and best_vendor and not in_combat:
                vendor_mode = True
                if min_v_dist < 4.0:
                    override_action = 8 # SELL
                else:
                    cmd_move = f"move_to:{best_vendor['x']:.2f}:{best_vendor['y']:.2f}:{best_vendor['z']:.2f}"
                    try: self.sock.sendall(f"{self.my_name}:{cmd_move}\n".encode('utf-8'))
                    except: pass
                    override_action = 1 
            
            # Prio 2: Kampf / Loot / Reise
            if not vendor_mode:
                if action == 8: override_action = 0 # Verbieten wenn nicht nötig
                
                # --- NEU: AGGRO CHECK (Das wolltest du!) ---
                # Wenn wir im Kampf sind, aber kein lebendes Ziel haben:
                if in_combat and not target_alive:
                    # Suche Mob, der MICH als Target hat (target != "0")
                    # (Wir gehen davon aus, dass ein Mob im Kampf mit Target ein Angreifer ist)
                    aggro_mob_guid = None
                    min_aggro_dist = 9999.0
                    
                    for mob in nearby_mobs:
                        # Hat der Mob ein Target? (GUID als String != "0")
                        if mob.get('target', '0') != '0' and mob.get('attackable', 0) == 1 and mob['hp'] > 0:
                            # Distanz checken
                            d = math.sqrt((mob['x']-me_pos['x'])**2 + (mob['y']-me_pos['y'])**2)
                            if d < min_aggro_dist:
                                min_aggro_dist = d
                                aggro_mob_guid = mob['guid']
                    
                    if aggro_mob_guid:
                        # Wir zwingen den Bot, diesen Mob ins Visier zu nehmen
                        # Wir senden Target GUID Command direkt hier, weil ActionSpace das nicht hergibt
                        try: self.sock.sendall(f"{self.my_name}:target_guid:{aggro_mob_guid}\n".encode('utf-8'))
                        except: pass
                        override_action = 0 # Warten bis Target gesetzt ist
                        # print(">>> AGGRO DETECTED! Wechsle Ziel... <<<")

                # Restliche Regeln
                elif is_casting and action in [1, 2, 3, 5, 6, 7]: override_action = 0 
                elif target_dead and dist_to_target < 3.0: override_action = 7 # Loot
                elif target_dead and dist_to_target >= 3.0: override_action = 1 # Hin
                elif target_alive and dist_to_target < 25.0:
                    if action == 1: override_action = 0 
                elif in_combat and target_alive and action == 4: override_action = 0
                elif action == 6 and hp_pct > 0.85: override_action = 0

        # --- COMMAND SENDEN ---
        cmd = ""
        if not self.last_state:
            override_action = 0
        if override_action == 8: # SELL
            if not self.last_state:
                override_action = 0
                cmd = ""
            else:
                v_guid = None
                min_v = 9999.0
                me_pos = {'x': self.last_state['x'], 'y': self.last_state['y'], 'z': self.last_state['z']}
                for guid, mob in self.npc_memory.items():
                    if mob.get('vendor', 0) == 1:
                        d = math.sqrt((mob['x']-me_pos['x'])**2 + (mob['y']-me_pos['y'])**2)
                        if d < min_v:
                            min_v = d
                            v_guid = guid
                if v_guid and min_v < 6.0: cmd = f"sell_grey:{v_guid}"
                else: override_action = 0

        elif override_action == 1: cmd = "move_forward:0"
        elif override_action == 2: cmd = "turn_left:0"
        elif override_action == 3: cmd = "turn_right:0"
        elif override_action == 4: 
            best_guid = None
            min_dist = 9999.0
            for mob in nearby_mobs:
                guid = mob['guid']
                if guid in self.blacklist: continue
                if mob.get('attackable', 0) == 0: continue
                if mob['hp'] == 0: continue 
                dx = mob['x'] - self.last_state['x']
                dy = mob['y'] - self.last_state['y']
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < min_dist:
                    min_dist = dist
                    best_guid = guid
            if best_guid: cmd = f"target_guid:{best_guid}"
            else: cmd = "" 

        elif override_action == 5: cmd = f"cast:{SPELL_SMITE}"
        elif override_action == 6: cmd = f"cast:{SPELL_HEAL}"
        elif override_action == 7: 
            loot_guid = None
            min_d = 6.0
            for mob in nearby_mobs:
                if mob['hp'] == 0:
                    d = math.sqrt((mob['x']-self.last_state['x'])**2 + (mob['y']-self.last_state['y'])**2)
                    if d < min_d:
                        min_d = d
                        loot_guid = mob['guid']
            if loot_guid:
                cmd = f"loot_guid:{loot_guid}"
                self.blacklist[loot_guid] = time.time() + self.BLACKLIST_DURATION
                if loot_guid in self.npc_memory: del self.npc_memory[loot_guid]
        
        if cmd:
            try: self.sock.sendall(f"{self.my_name}:{cmd}\n".encode('utf-8'))
            except: pass

        time.sleep(0.3)
        data = self._get_state_from_server()
        if not data: return np.zeros(10), 0, True, False, {}

        # --- MEMORY UPDATE ---
        current_mobs = data.get('nearby_mobs', [])
        discovery_reward = 0.0
        for mob in current_mobs:
            guid = mob['guid']
            if mob['hp'] == 0: 
                if guid not in self.blacklist:
                    self.blacklist[guid] = time.time() + self.BLACKLIST_DURATION
                    if guid in self.npc_memory: del self.npc_memory[guid]
                continue
            if guid in self.blacklist: continue
            if guid not in self.npc_memory:
                self.npc_memory[guid] = mob
                discovery_reward += 0.5 
            else: self.npc_memory[guid] = mob

        max_hp = max(1, data['max_hp'])
        hp_pct = data['hp'] / max_hp
        mana_pct = data['power'] / max(1, data['max_power'])
        xp_gained = data.get('xp_gained', 0)
        loot_money = data.get('loot_copper', 0)
        loot_score = data.get('loot_score', 0)
        free_slots_curr = data.get('free_slots', 0)
        
        t_hp, t_exists, dist_norm, angle_norm = 0.0, 0.0, 0.0, 0.0
        if data['target_status'] == 'alive': 
            t_hp = data['target_hp'] / 100.0
            t_exists = 1.0
            dx = data['tx'] - data['x']
            dy = data['ty'] - data['y']
            dist = math.sqrt(dx*dx + dy*dy)
            dist_norm = min(dist, 40.0) / 40.0
            target_angle = math.atan2(dy, dx)
            rel = target_angle - data['o']
            while rel > math.pi: rel -= 2*math.pi
            while rel < -math.pi: rel += 2*math.pi
            angle_norm = rel / math.pi

        is_casting = 1.0 if data.get('casting') == 'true' else 0.0
        in_combat = 1.0 if data['combat'] == 'true' else 0.0
        slots_norm = free_slots_curr / 20.0 

        obs = np.array([hp_pct, mana_pct, t_hp, t_exists, in_combat, dist_norm, angle_norm, is_casting, 0.0, slots_norm], dtype=np.float32)

        # --- REWARDS ---
        reward = -0.01 + discovery_reward

        # 1. NEU: Upgrade Reward
        if data.get('equipped_upgrade') == 'true':
            reward += 100.0 # Bessere Rüstung ist super!
            print(">>> UPGRADE EQUIPPED! +100 <<<")

        if data.get('leveled_up') == 'true':
            reward += 2000.0
            terminated = True
        elif xp_gained > 0:
            reward += 100.0 + (xp_gained * 2.0)
            print(f">>> KILL! XP: {xp_gained} <<<")
            
        if loot_money > 0 or loot_score > 0:
            reward += (loot_money * 0.1) + (loot_score * 2.0)
            print(f">>> LOOT! Copper: {loot_money}, ItemScore: {loot_score} <<<")

        if self.last_state and free_slots_curr > self.last_state.get('free_slots', 0):
            reward += 50.0 
            print(">>> SOLD ITEMS! <<<")

        if override_action == 5: 
            if t_exists > 0.5: reward += 0.5
            else: reward -= 0.1

        elif override_action == 6: 
            # 2. NEU: Heil-Reward entfernt
            if hp_pct > 0.8: 
                reward -= 0.5 # Nur Strafe bei Unsinn
            
        if in_combat and t_exists > 0.5:
            if override_action in [1, 2, 3]: 
                reward -= 0.5 
                if abs(angle_norm) > 0.2 and override_action in [2, 3]:
                    reward += 0.6 

        terminated = False
        if data['hp'] == 0:
            reward -= 100.0
            terminated = True
        elif mana_pct < 0.05:
            reward -= 10.0
            terminated = True
        else:
            if not terminated: terminated = False

        self.last_state = data
        return obs, reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        print(">>> RESETTING ENVIRONMENT... <<<")
        
        # Versuche Reset-Befehl zu senden
        name_to_use = self.my_name or self.bot_name
        if name_to_use:
            try:
                self.sock.sendall(f"{name_to_use}:reset:0\n".encode('utf-8'))
            except: pass
        
        # WARTE AUF DATEN (Zwingend!)
        data = None
        start_time = time.time()
        while data is None:
            data = self._get_state_from_server()
            if data is None:
                if time.time() - start_time > 10.0:
                    print("Reset Timeout: keine Bot-Daten, versuche Fallback-Spieler.")
                    data = self._get_state_from_server(allow_any_player=True)
                    break
                print("Warte auf Server-Antwort für Reset...")
                time.sleep(1.0)
        
        self.last_state = data
        
        # Initial Observation
        obs = np.zeros(10, dtype=np.float32)
        # Wir können obs auch direkt aus data füllen, wenn wir wollen, 
        # aber Nullen sind okay für den Start.
        
        return obs, {}

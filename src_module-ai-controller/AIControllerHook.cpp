#include "ScriptMgr.h"
#include "Player.h"
#include "Config.h"
#include "Chat.h"
#include "Log.h"
#include "World.h"
#include "WorldSession.h"
#include "WorldSessionMgr.h"
#include "WorldSocket.h" 
#include "ObjectAccessor.h"
#include <cmath>
#include <boost/asio.hpp>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <vector>
#include <string>
#include <queue>
#include <sstream>
#include <unordered_map>
#include "GameTime.h" 
#include <atomic>
#include "GridNotifiers.h"
#include "GridNotifiersImpl.h"
#include "CellImpl.h"
#include "Cell.h"
#include "Item.h"   
#include "ItemTemplate.h"
#include "Bag.h"
#include "AccountMgr.h"
#include "DatabaseEnv.h"
#include "PreparedStatement.h"
#include "QueryHolder.h"
#include "DatabaseWorkerPool.h"
#include "AsyncCallbackProcessor.h"
#include "ObjectMgr.h"

// WICHTIG: Zuerst MySQLConnection, dann CharacterDatabase
#include "MySQLConnection.h"

// Geht auch so
#include "CharacterDatabase.h"

using boost::asio::ip::tcp;
using CharacterDatabasePreparedStatement = PreparedStatement<CharacterDatabaseConnection>;

// HINWEIS: enum PlayerLoginQueryIndex wurde entfernt, da es bereits in Player.h definiert ist.

// --- Bot Login Helper (Nachbau von LoginQueryHolder aus CharacterHandler.cpp) ---
// WICHTIG: Async-Holder wie im Core-Login, damit ASYNC PreparedStatements genutzt werden.
class BotLoginQueryHolder : public CharacterDatabaseQueryHolder {
private:
    uint32 m_accountId;
    ObjectGuid m_guid;
public:
    BotLoginQueryHolder(uint32 accountId, ObjectGuid guid)
        : m_accountId(accountId), m_guid(guid) {
    }

    // Gibt die GUID zurück (wichtig für HandlePlayerLoginFromDB)
    ObjectGuid GetGuid() const { return m_guid; }

    bool Initialize() {
        SetSize(MAX_PLAYER_LOGIN_QUERY);

        bool res = true;
        ObjectGuid::LowType lowGuid = m_guid.GetCounter();

        CharacterDatabasePreparedStatement* stmt = CharacterDatabase.GetPreparedStatement(CHAR_SEL_CHARACTER);
        stmt->SetData(0, lowGuid);
        res &= SetPreparedQuery(PLAYER_LOGIN_QUERY_LOAD_FROM, stmt);

        stmt = CharacterDatabase.GetPreparedStatement(CHAR_SEL_CHARACTER_AURAS);
        stmt->SetData(0, lowGuid);
        res &= SetPreparedQuery(PLAYER_LOGIN_QUERY_LOAD_AURAS, stmt);

        stmt = CharacterDatabase.GetPreparedStatement(CHAR_SEL_CHARACTER_SPELL);
        stmt->SetData(0, lowGuid);
        res &= SetPreparedQuery(PLAYER_LOGIN_QUERY_LOAD_SPELLS, stmt);

        stmt = CharacterDatabase.GetPreparedStatement(CHAR_SEL_CHARACTER_QUESTSTATUS);
        stmt->SetData(0, lowGuid);
        res &= SetPreparedQuery(PLAYER_LOGIN_QUERY_LOAD_QUEST_STATUS, stmt);

        stmt = CharacterDatabase.GetPreparedStatement(CHAR_SEL_CHARACTER_DAILYQUESTSTATUS);
        stmt->SetData(0, lowGuid);
        res &= SetPreparedQuery(PLAYER_LOGIN_QUERY_LOAD_DAILY_QUEST_STATUS, stmt);

        stmt = CharacterDatabase.GetPreparedStatement(CHAR_SEL_CHARACTER_WEEKLYQUESTSTATUS);
        stmt->SetData(0, lowGuid);
        res &= SetPreparedQuery(PLAYER_LOGIN_QUERY_LOAD_WEEKLY_QUEST_STATUS, stmt);

        stmt = CharacterDatabase.GetPreparedStatement(CHAR_SEL_CHARACTER_MONTHLYQUESTSTATUS);
        stmt->SetData(0, lowGuid);
        res &= SetPreparedQuery(PLAYER_LOGIN_QUERY_LOAD_MONTHLY_QUEST_STATUS, stmt);

        stmt = CharacterDatabase.GetPreparedStatement(CHAR_SEL_CHARACTER_SEASONALQUESTSTATUS);
        stmt->SetData(0, lowGuid);
        res &= SetPreparedQuery(PLAYER_LOGIN_QUERY_LOAD_SEASONAL_QUEST_STATUS, stmt);

        stmt = CharacterDatabase.GetPreparedStatement(CHAR_SEL_CHARACTER_REPUTATION);
        stmt->SetData(0, lowGuid);
        res &= SetPreparedQuery(PLAYER_LOGIN_QUERY_LOAD_REPUTATION, stmt);

        stmt = CharacterDatabase.GetPreparedStatement(CHAR_SEL_CHARACTER_INVENTORY);
        stmt->SetData(0, lowGuid);
        res &= SetPreparedQuery(PLAYER_LOGIN_QUERY_LOAD_INVENTORY, stmt);

        stmt = CharacterDatabase.GetPreparedStatement(CHAR_SEL_CHARACTER_ACTIONS);
        stmt->SetData(0, lowGuid);
        res &= SetPreparedQuery(PLAYER_LOGIN_QUERY_LOAD_ACTIONS, stmt);

        stmt = CharacterDatabase.GetPreparedStatement(CHAR_SEL_CHARACTER_SKILLS);
        stmt->SetData(0, lowGuid);
        res &= SetPreparedQuery(PLAYER_LOGIN_QUERY_LOAD_SKILLS, stmt);

        stmt = CharacterDatabase.GetPreparedStatement(CHAR_SEL_CHARACTER_EQUIPMENTSETS);
        stmt->SetData(0, lowGuid);
        res &= SetPreparedQuery(PLAYER_LOGIN_QUERY_LOAD_EQUIPMENT_SETS, stmt);

        stmt = CharacterDatabase.GetPreparedStatement(CHAR_SEL_CHARACTER_GLYPHS);
        stmt->SetData(0, lowGuid);
        res &= SetPreparedQuery(PLAYER_LOGIN_QUERY_LOAD_GLYPHS, stmt);

        stmt = CharacterDatabase.GetPreparedStatement(CHAR_SEL_CHARACTER_TALENTS);
        stmt->SetData(0, lowGuid);
        res &= SetPreparedQuery(PLAYER_LOGIN_QUERY_LOAD_TALENTS, stmt);

        // Weitere (weniger kritisch, aber gut für Vollständigkeit):
        stmt = CharacterDatabase.GetPreparedStatement(CHAR_SEL_CHARACTER_HOMEBIND);
        stmt->SetData(0, lowGuid);
        res &= SetPreparedQuery(PLAYER_LOGIN_QUERY_LOAD_HOME_BIND, stmt);
        // LoadQuery(CHAR_SEL_CHARACTER_SPELLCOOLDOWNS, PLAYER_LOGIN_QUERY_LOAD_SPELL_COOLDOWNS);

        /*
        if (sWorld->getBoolConfig(CONFIG_DECLINED_NAMES_USED))
            LoadQuery(CHAR_SEL_CHARACTER_DECLINEDNAMES, PLAYER_LOGIN_QUERY_LOAD_DECLINED_NAMES);
        */

        // LoadQuery(CHAR_SEL_CHARACTER_ACHIEVEMENTS, PLAYER_LOGIN_QUERY_LOAD_ACHIEVEMENTS);
        // LoadQuery(CHAR_SEL_CHARACTER_CRITERIAPROGRESS, PLAYER_LOGIN_QUERY_LOAD_CRITERIA_PROGRESS);
        // LoadQuery(CHAR_SEL_CHARACTER_ENTRY_POINT, PLAYER_LOGIN_QUERY_LOAD_ENTRY_POINT);
        // LoadQuery(CHAR_SEL_ACCOUNT_DATA, PLAYER_LOGIN_QUERY_LOAD_ACCOUNT_DATA);
        // LoadQuery(CHAR_SEL_CHARACTER_RANDOMBG, PLAYER_LOGIN_QUERY_LOAD_RANDOM_BG);
        // LoadQuery(CHAR_SEL_CHARACTER_BANNED, PLAYER_LOGIN_QUERY_LOAD_BANNED);
        // LoadQuery(CHAR_SEL_CHARACTER_QUESTSTATUSREW, PLAYER_LOGIN_QUERY_LOAD_QUEST_STATUS_REW);
        // LoadQuery(CHAR_SEL_BREW_OF_THE_MONTH, PLAYER_LOGIN_QUERY_LOAD_BREW_OF_THE_MONTH);

        // LoadQuery(CHAR_SEL_CORPSE_LOCATION, PLAYER_LOGIN_QUERY_LOAD_CORPSE_LOCATION);
        // LoadQuery(CHAR_SEL_CHAR_SETTINGS, PLAYER_LOGIN_QUERY_LOAD_CHARACTER_SETTINGS);
        // LoadQuery(CHAR_SEL_CHAR_PETS, PLAYER_LOGIN_QUERY_LOAD_PET_SLOTS);
        // LoadQuery(CHAR_SEL_CHAR_ACHIEVEMENT_OFFLINE_UPDATES, PLAYER_LOGIN_QUERY_LOAD_OFFLINE_ACHIEVEMENTS_UPDATES);

        return res;
    }
};

struct AICommand {
    std::string playerName;
    std::string actionType;
    std::string value;
};

std::mutex g_Mutex;
std::string g_CurrentJsonState = "{}";
bool g_HasNewState = false;
std::atomic<uint64_t> g_StateVersion{ 0 };
std::queue<AICommand> g_CommandQueue;

struct AIPlayerEvents
{
    long long xp_gained = 0;
    long long loot_copper = 0;
    long long loot_score = 0;
    bool leveled_up = false;
    bool equipped_upgrade = false;

    bool IsEmpty() const
    {
        return xp_gained == 0 && loot_copper == 0 && loot_score == 0 && !leveled_up && !equipped_upgrade;
    }
};

std::mutex g_EventMutex;
std::unordered_map<uint64, AIPlayerEvents> g_PlayerEvents;
AsyncCallbackProcessor<SQLQueryHolderCallback> g_QueryHolderProcessor;
std::unordered_map<uint32, WorldSession*> g_BotSessions;
std::mutex g_BotSessionsMutex;

// --- HELPER ---

int GetItemScore(ItemTemplate const* proto) {
    if (!proto) return 0;
    int score = 0;
    score += proto->Quality * 10;
    score += proto->ItemLevel;
    score += proto->Armor;
    if (proto->Class == ITEM_CLASS_WEAPON) {
        score += (int)(proto->Damage[0].DamageMax + proto->Damage[0].DamageMin);
    }
    for (int i = 0; i < proto->StatsCount; ++i) {
        score += proto->ItemStat[i].ItemStatValue * 2;
    }
    return score;
}

static bool IsBotControlledPlayer(Player* p)
{
    if (!p)
        return false;

    WorldSession* sess = p->GetSession();
    if (!sess)
        return false;

    std::lock_guard<std::mutex> lock(g_BotSessionsMutex);
    for (auto const& it : g_BotSessions)
        if (it.second == sess)
            return true;

    return false;
}


// --- PER-PLAYER EVENT HELPERS ---
static inline uint64 AIEventKey(Player* player)
{
    return player ? player->GetGUID().GetRawValue() : 0ULL;
}

static void AddXPGained(Player* player, uint32 amount)
{
    if (!player || amount == 0)
        return;

    std::lock_guard<std::mutex> lock(g_EventMutex);
    g_PlayerEvents[AIEventKey(player)].xp_gained += amount;
}

static void AddLootCopper(Player* player, uint32 amount)
{
    if (!player || amount == 0)
        return;

    std::lock_guard<std::mutex> lock(g_EventMutex);
    g_PlayerEvents[AIEventKey(player)].loot_copper += amount;
}

static void AddLootScore(Player* player, uint32 amount)
{
    if (!player || amount == 0)
        return;

    std::lock_guard<std::mutex> lock(g_EventMutex);
    g_PlayerEvents[AIEventKey(player)].loot_score += amount;
}

static void SetLeveledUp(Player* player)
{
    if (!player)
        return;

    std::lock_guard<std::mutex> lock(g_EventMutex);
    g_PlayerEvents[AIEventKey(player)].leveled_up = true;
}

static void SetEquippedUpgrade(Player* player)
{
    if (!player)
        return;

    std::lock_guard<std::mutex> lock(g_EventMutex);
    g_PlayerEvents[AIEventKey(player)].equipped_upgrade = true;
}

// Atomisch: Events für genau diesen Player holen und dabei resetten.
static AIPlayerEvents ConsumePlayerEvents(Player* player)
{
    AIPlayerEvents out;
    if (!player)
        return out;

    uint64 key = AIEventKey(player);
    std::lock_guard<std::mutex> lock(g_EventMutex);

    auto it = g_PlayerEvents.find(key);
    if (it == g_PlayerEvents.end())
        return out;

    out = it->second;
    g_PlayerEvents.erase(it);
    return out;
}

void TryEquipIfBetter(Player* player, uint16 srcPos) {
    uint8 bag = srcPos >> 8;
    uint8 slot = srcPos & 255;
    Item* newItem = player->GetItemByPos(bag, slot);
    if (!newItem) return;
    if (player->CanUseItem(newItem) != EQUIP_ERR_OK) return;

    ItemTemplate const* proto = newItem->GetTemplate();
    uint16 destSlot = 0xffff;

    switch (proto->InventoryType) {
    case INVTYPE_HEAD: destSlot = EQUIPMENT_SLOT_HEAD; break;
    case INVTYPE_SHOULDERS: destSlot = EQUIPMENT_SLOT_SHOULDERS; break;
    case INVTYPE_BODY: case INVTYPE_CHEST: case INVTYPE_ROBE: destSlot = EQUIPMENT_SLOT_CHEST; break;
    case INVTYPE_WAIST: destSlot = EQUIPMENT_SLOT_WAIST; break;
    case INVTYPE_LEGS: destSlot = EQUIPMENT_SLOT_LEGS; break;
    case INVTYPE_FEET: destSlot = EQUIPMENT_SLOT_FEET; break;
    case INVTYPE_WRISTS: destSlot = EQUIPMENT_SLOT_WRISTS; break;
    case INVTYPE_HANDS: destSlot = EQUIPMENT_SLOT_HANDS; break;
    case INVTYPE_WEAPON: case INVTYPE_2HWEAPON: case INVTYPE_WEAPONMAINHAND: destSlot = EQUIPMENT_SLOT_MAINHAND; break;
    case INVTYPE_SHIELD: case INVTYPE_WEAPONOFFHAND: destSlot = EQUIPMENT_SLOT_OFFHAND; break;
    }

    if (destSlot != 0xffff) {
        int newScore = GetItemScore(proto);
        int currentScore = -1;
        Item* currentItem = player->GetItemByPos(INVENTORY_SLOT_BAG_0, destSlot);
        if (currentItem) currentScore = GetItemScore(currentItem->GetTemplate());

        if (newScore > currentScore) {
            player->SwapItem(srcPos, destSlot);
            player->PlayDistanceSound(120, player);
            SetEquippedUpgrade(player);
            LOG_INFO("module", "AI-GEAR: Upgrade angelegt! (Slot: {})", destSlot);
        }
    }
}

class CreatureCollector {
public:
    std::vector<Creature*> foundCreatures;
    Player* i_player;
    CreatureCollector(Player* player) : i_player(player) {}

    void Visit(CreatureMapType& m) {
        for (CreatureMapType::iterator itr = m.begin(); itr != m.end(); ++itr) {
            Creature* creature = itr->GetSource();
            if (creature) {
                if (!creature->IsInWorld()) continue;
                if (creature->IsTotem() || creature->IsPet()) continue;
                if (creature->GetCreatureTemplate()->type == CREATURE_TYPE_CRITTER) continue;

                if (!creature->IsAlive()) {
                    if (i_player->GetDistance(creature) > 10.0f) continue;
                }
                else if (i_player->GetDistance(creature) > 50.0f) continue;

                foundCreatures.push_back(creature);
            }
        }
    }
    template<class SKIP> void Visit(GridRefMgr<SKIP>&) {}
};

uint32 GetFreeBagSlots(Player* player) {
    uint32 freeSlots = 0;
    for (uint8 slot = INVENTORY_SLOT_ITEM_START; slot < INVENTORY_SLOT_ITEM_END; ++slot) {
        if (!player->GetItemByPos(INVENTORY_SLOT_BAG_0, slot)) freeSlots++;
    }
    for (uint8 bag = INVENTORY_SLOT_BAG_START; bag < INVENTORY_SLOT_BAG_END; ++bag) {
        Bag* bagItem = (Bag*)player->GetItemByPos(INVENTORY_SLOT_BAG_0, bag);
        if (bagItem) freeSlots += bagItem->GetFreeSlots();
    }
    return freeSlots;
}

// --- SERVER THREAD ---
//
// Multi-Client: Thread pro Client (synchrones IO). Jeder Client bekommt den State-Stream
// unabhängig (kein globales "g_HasNewState=false" mehr).
//
// State-Push wird über g_StateVersion getriggert: OnUpdate inkrementiert die Version,
// jeder Client merkt sich seine letzte gesehene Version.
//
static void HandleAIClient(tcp::socket socket)
{
    try
    {
        LOG_INFO("module", ">>> CLIENT VERBUNDEN! <<<");

        // optional: kleine Latenz-Optimierung
        boost::system::error_code ec;
        socket.set_option(tcp::no_delay(true), ec);

        char data_[8192];
        std::string incomingBuffer;

        uint64_t lastVersion = 0;
        auto lastSend = std::chrono::steady_clock::now();

        // Initial-State sofort senden
        {
            std::string initial;
            {
                std::lock_guard<std::mutex> lock(g_Mutex);
                initial = g_CurrentJsonState + "\n";
            }
            boost::asio::write(socket, boost::asio::buffer(initial));
            lastVersion = g_StateVersion.load(std::memory_order_relaxed);
            lastSend = std::chrono::steady_clock::now();
        }

        while (true)
        {
            // 1) State senden, wenn neue Version oder Keepalive (alle 500ms)
            std::string msg;
            uint64_t v = g_StateVersion.load(std::memory_order_relaxed);

            if (v != lastVersion)
            {
                std::lock_guard<std::mutex> lock(g_Mutex);
                msg = g_CurrentJsonState + "\n";
                lastVersion = v;
            }
            else
            {
                auto now = std::chrono::steady_clock::now();
                if (now - lastSend >= std::chrono::milliseconds(500))
                {
                    std::lock_guard<std::mutex> lock(g_Mutex);
                    msg = g_CurrentJsonState + "\n";
                }
            }

            if (!msg.empty())
            {
                boost::asio::write(socket, boost::asio::buffer(msg));
                lastSend = std::chrono::steady_clock::now();
            }

            // 2) Eingehende Commands lesen (non-blocking-ish über available())
            if (socket.available() > 0)
            {
                boost::system::error_code error;
                size_t length = socket.read_some(boost::asio::buffer(data_), error);

                if (error == boost::asio::error::eof)
                    break; // client disconnected
                if (error)
                    throw boost::system::system_error(error);

                incomingBuffer.append(data_, length);

                size_t newlinePos = 0;
                while ((newlinePos = incomingBuffer.find('\n')) != std::string::npos)
                {
                    std::string line = incomingBuffer.substr(0, newlinePos);
                    incomingBuffer.erase(0, newlinePos + 1);
                    if (line.empty())
                        continue;

                    // Format: playerName:actionType:value
                    size_t p1 = line.find(':');
                    size_t p2 = line.find(':', p1 + 1);
                    if (p1 != std::string::npos && p2 != std::string::npos)
                    {
                        AICommand cmd;
                        cmd.playerName = line.substr(0, p1);
                        cmd.actionType = line.substr(p1 + 1, p2 - p1 - 1);
                        cmd.value = line.substr(p2 + 1);

                        std::lock_guard<std::mutex> lock(g_Mutex);
                        g_CommandQueue.push(cmd);
                    }
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("module", "Verbindung verloren: {}", e.what());
    }

    LOG_INFO("module", ">>> CLIENT GETRENNT <<<");
}

void AIServerThread()
{
    try
    {
        boost::asio::io_context io_context;
        tcp::acceptor acceptor(io_context, tcp::endpoint(tcp::v4(), 5000));
        LOG_INFO("module", ">>> AI-SOCKET: Lausche auf Port 5000... <<<");

        while (true)
        {
            tcp::socket socket(io_context);
            acceptor.accept(socket);

            // Thread pro Client
            std::thread(&HandleAIClient, std::move(socket)).detach();
        }
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("module", "Server Crash: {}", e.what());
    }
}

// --- LOGIC ---
class AIControllerWorldScript : public WorldScript {
private:
    uint32 _fastTimer;
    uint32 _slowTimer;
    uint32 _faceTimer;
    std::unordered_map<std::string, std::string> _cachedNearbyMobsPerPlayer;
    void CollectOnlinePlayers(std::vector<Player*>& players) {
        std::shared_lock lock(*HashMapHolder<Player>::GetLock());
        players.reserve(ObjectAccessor::GetPlayers().size());
        for (auto const& it : ObjectAccessor::GetPlayers()) {
            Player* player = it.second;
            if (!player || !player->IsInWorld()) {
                continue;
            }
            players.push_back(player);
        }
    }
public:
    AIControllerWorldScript() : WorldScript("AIControllerWorldScript"), _fastTimer(0), _slowTimer(0), _faceTimer(0) {}
    void OnStartup() override { std::thread(AIServerThread).detach(); }

    void OnUpdate(uint32 diff) override {
        _fastTimer += diff; _slowTimer += diff; _faceTimer += diff;

        if (_faceTimer >= 150) {
            _faceTimer = 0;
            auto const& sessions = sWorldSessionMgr->GetAllSessions();
            for (auto const& pair : sessions) {
                Player* p = pair.second->GetPlayer();
                if (!p) continue;
                if (p->IsInCombat() || p->HasUnitState(UNIT_STATE_CASTING)) {
                    Unit* target = p->GetSelectedUnit();
                    if (target) p->SetFacingToObject(target);
                }
            }
        }

        g_QueryHolderProcessor.ProcessReadyCallbacks();

        {
            std::lock_guard<std::mutex> lock(g_Mutex);
            while (!g_CommandQueue.empty()) {
                AICommand cmd = g_CommandQueue.front();
                g_CommandQueue.pop();
                Player* player = ObjectAccessor::FindPlayerByName(cmd.playerName);
                if (!player) continue;
                if (!IsBotControlledPlayer(player)) continue;
                if (cmd.actionType == "say") player->Say(cmd.value, LANG_UNIVERSAL);
                else if (cmd.actionType == "stop") { player->GetMotionMaster()->Clear(); player->GetMotionMaster()->MoveIdle(); }
                else if (cmd.actionType == "turn_left" || cmd.actionType == "turn_right") {
                    float o = player->GetOrientation();
                    float step = (cmd.actionType == "turn_left") ? 0.5f : -0.5f;
                    o += step; if (o > 6.283f) o -= 6.283f; if (o < 0) o += 6.283f;
                    player->SetFacingTo(o);
                }
                else if (cmd.actionType == "move_forward") {
                    float o = player->GetOrientation();
                    float x = player->GetPositionX() + (3.0f * std::cos(o));
                    float y = player->GetPositionY() + (3.0f * std::sin(o));
                    float z = player->GetPositionZ();
                    player->UpdateGroundPositionZ(x, y, z);
                    player->GetMotionMaster()->MovePoint(1, x, y, z);
                }
                else if (cmd.actionType == "target_nearest") {
                    float range = 30.0f;
                    if (!cmd.value.empty()) {
                        try {
                            float parsed = std::stof(cmd.value);
                            if (parsed > 0.0f) range = parsed;
                        }
                        catch (std::exception const&) {
                        }
                    }
                    Unit* target = player->SelectNearbyTarget(nullptr, range);
                    if (target && player->IsValidAttackTarget(target)) {
                        player->SetSelection(target->GetGUID());
                        player->SetTarget(target->GetGUID());
                        player->SetFacingToObject(target);
                    }
                }
                else if (cmd.actionType == "cast") {
                    uint32 spellId = std::stoi(cmd.value);
                    Unit* target = player->GetSelectedUnit();
                    // Self-target spells: Lesser Heal (2050), Power Word: Shield (17)
                    if (spellId == 2050 || spellId == 17) target = player;
                    // Enemy-target spells: Smite (585), Shadow Word: Pain (589)
                    else if (spellId == 585 || spellId == 589) {
                        if (!target || target == player) {
                            target = player->SelectNearbyTarget(nullptr, 30.0f);
                            if (target && !player->IsValidAttackTarget(target)) target = nullptr;
                        }
                    }
                    else if (!target) target = player;
                    if (target) {
                        bool isOffensiveOnSelf = ((spellId == 585 || spellId == 589) && target == player);
                        if (!isOffensiveOnSelf) player->CastSpell(target, spellId, false);
                    }
                }
                else if (cmd.actionType == "reset") {
                    player->CombatStop(true); player->AttackStop(); player->GetMotionMaster()->Clear();
                    if (!player->isDead()) { player->ResurrectPlayer(1.0f, false); player->SpawnCorpseBones(); }
                    player->SetHealth(player->GetMaxHealth()); player->SetPower(player->getPowerType(), player->GetMaxPower(player->getPowerType()));
                    player->RemoveAllSpellCooldown(); player->RemoveAllAuras();
                    player->TeleportTo(player->m_homebindMapId, player->m_homebindX, player->m_homebindY, player->m_homebindZ, player->GetOrientation());
                }
                else if (cmd.actionType == "move_to") {
                    std::string val = cmd.value;
                    size_t p1 = val.find(':');
                    size_t p2 = val.find(':', p1 + 1);
                    if (p1 != std::string::npos && p2 != std::string::npos) {
                        float tx = std::stof(val.substr(0, p1)); float ty = std::stof(val.substr(p1 + 1, p2 - p1 - 1)); float tz = std::stof(val.substr(p2 + 1));
                        player->UpdateGroundPositionZ(tx, ty, tz);
                        Position pos(tx, ty, tz, 0.0f);
                        player->GetMotionMaster()->MovePoint(1, pos, (ForcedMovement)0, 0.0f, true);
                    }
                }
                else if (cmd.actionType == "target_guid") {
                    ObjectGuid guid = ObjectGuid(std::stoull(cmd.value));
                    Unit* target = ObjectAccessor::GetUnit(*player, guid);
                    if (target) { player->SetSelection(target->GetGUID()); player->SetTarget(target->GetGUID()); player->SetFacingToObject(target); player->AttackStop(); }
                }
                else if (cmd.actionType == "loot_guid") {
                    ObjectGuid guid = ObjectGuid(std::stoull(cmd.value));
                    Creature* target = ObjectAccessor::GetCreature(*player, guid);
                    if (target && target->isDead()) {
                        if (player->GetDistance(target) <= 10.0f) {
                            player->SendLoot(target->GetGUID(), LOOT_CORPSE);
                            Loot* loot = &target->loot;
                            uint32 gold = loot->gold;
                            if (gold > 0) {
                                loot->gold = 0; player->ModifyMoney(gold);
                                player->UpdateAchievementCriteria(ACHIEVEMENT_CRITERIA_TYPE_LOOT_MONEY, gold);
                                WorldPacket data(SMSG_LOOT_MONEY_NOTIFY, 4 + 1); data << uint32(gold); data << uint8(1); player->GetSession()->SendPacket(&data);
                                AddLootCopper(player, gold);
                            }
                            for (uint8 i = 0; i < loot->items.size(); ++i) {
                                LootItem* item = loot->LootItemInSlot(i, player);
                                if (item && !item->is_looted && !item->freeforall && !item->needs_quest) {
                                    ItemPosCountVec dest;
                                    InventoryResult msg = player->CanStoreNewItem(NULL_BAG, NULL_SLOT, dest, item->itemid, item->count);
                                    if (msg == EQUIP_ERR_OK) {
                                        Item* newItem = player->StoreNewItem(dest, item->itemid, true);
                                        item->count = 0; item->is_looted = true;
                                        if (newItem) player->SendNewItem(newItem, 1, false, true);
                                        TryEquipIfBetter(player, dest[0].pos);
                                        ItemTemplate const* proto = sObjectMgr->GetItemTemplate(item->itemid);
                                        if (proto) { AddLootScore(player, 1); }
                                    }
                                }
                            }
                            target->RemoveFlag(UNIT_DYNAMIC_FLAGS, UNIT_DYNFLAG_LOOTABLE);
                            target->AllLootRemovedFromCorpse();
                            player->SendLootRelease(player->GetLootGUID());
                            player->SetSelection(ObjectGuid::Empty); player->SetTarget(ObjectGuid::Empty); player->AttackStop();
                        }
                    }
                }
                else if (cmd.actionType == "sell_grey") {
                    ObjectGuid guid = ObjectGuid(std::stoull(cmd.value));
                    Creature* vendor = ObjectAccessor::GetCreature(*player, guid);
                    if (vendor && player->GetDistance(vendor) <= 15.0f) {
                        player->StopMoving();
                        uint32 totalMoney = 0;
                        for (uint8 i = INVENTORY_SLOT_ITEM_START; i < INVENTORY_SLOT_ITEM_END; ++i) {
                            if (Item* item = player->GetItemByPos(INVENTORY_SLOT_BAG_0, i)) {
                                ItemTemplate const* proto = item->GetTemplate();
                                if (proto->SellPrice > 0 && proto->ItemId != 6948) {
                                    uint32 price = proto->SellPrice * item->GetCount();
                                    totalMoney += price; player->DestroyItem(INVENTORY_SLOT_BAG_0, i, true);
                                }
                            }
                        }
                        for (uint8 bag = INVENTORY_SLOT_BAG_START; bag < INVENTORY_SLOT_BAG_END; ++bag) {
                            if (Bag* bagItem = (Bag*)player->GetItemByPos(INVENTORY_SLOT_BAG_0, bag)) {
                                for (uint8 i = 0; i < bagItem->GetBagSize(); ++i) {
                                    if (Item* item = bagItem->GetItemByPos(i)) {
                                        ItemTemplate const* proto = item->GetTemplate();
                                        if (proto->SellPrice > 0 && proto->ItemId != 6948) {
                                            uint32 price = proto->SellPrice * item->GetCount();
                                            totalMoney += price; player->DestroyItem(bag, i, true);
                                        }
                                    }
                                }
                            }
                        }
                        if (totalMoney > 0) {
                            player->ModifyMoney(totalMoney);
                            player->PlayDistanceSound(120, player);
                            AddLootCopper(player, totalMoney);
                        }
                        player->SetSelection(ObjectGuid::Empty); player->SetTarget(ObjectGuid::Empty);
                    }
                }
            }
        }

        if (_fastTimer >= 400) {
            _fastTimer = 0;
            std::stringstream ss;
            ss << "{ \"players\": [";
            bool first = true;
            std::vector<Player*> players;
            CollectOnlinePlayers(players);
            for (Player* p : players) {
                if (!p) continue;
                if (!first) ss << ", ";
                first = false;
                ss << "{";
                ss << "\"name\": \"" << p->GetName() << "\", ";
                ss << "\"hp\": " << p->GetHealth() << ", ";
                ss << "\"max_hp\": " << p->GetMaxHealth() << ", ";
                ss << "\"power\": " << p->GetPower(p->getPowerType()) << ", ";
                ss << "\"max_power\": " << p->GetMaxPower(p->getPowerType()) << ", ";
                ss << "\"level\": " << (int)p->GetLevel() << ", ";
                ss << "\"x\": " << p->GetPositionX() << ", ";
                ss << "\"y\": " << p->GetPositionY() << ", ";
                ss << "\"z\": " << p->GetPositionZ() << ", ";
                ss << "\"o\": " << p->GetOrientation() << ", ";
                ss << "\"combat\": \"" << (p->IsInCombat() ? "true" : "false") << "\", ";
                ss << "\"casting\": \"" << (p->HasUnitState(UNIT_STATE_CASTING) ? "true" : "false") << "\", ";
                ss << "\"free_slots\": " << GetFreeBagSlots(p) << ", ";
                AIPlayerEvents ev = ConsumePlayerEvents(p);

                ss << "\"equipped_upgrade\": \"" << (ev.equipped_upgrade ? "true" : "false") << "\", ";
                Unit* target = p->GetSelectedUnit();
                std::string tStatus = "none"; uint32 tHp = 0; uint32 tLevel = 0; float tx = 0, ty = 0, tz = 0;
                if (target) {
                    tStatus = target->IsAlive() ? "alive" : "dead";
                    tHp = target->GetHealth();
                    tLevel = target->GetLevel();
                    tx = target->GetPositionX(); ty = target->GetPositionY(); tz = target->GetPositionZ();
                }
                // Aura states for priest spells (all ranks checked)
                // Helper lambda: check if player has any rank of a spell family
                auto hasAnyRank = [](Unit* u, std::initializer_list<uint32> ids) -> bool {
                    for (uint32 id : ids) { if (u->HasAura(id)) return true; }
                    return false;
                };

                // Player buffs
                bool hasShield = hasAnyRank(p, {17, 592, 600, 3747, 6065, 6066, 10898, 10899, 10900, 10901, 25217, 25218, 48065, 48066}); // PW:Shield all ranks
                bool hasRenew = hasAnyRank(p, {139, 6074, 6075, 6076, 6077, 6078, 10927, 10928, 10929, 25315, 25221, 25222, 48067, 48068}); // Renew all ranks
                bool hasInnerFire = hasAnyRank(p, {588, 7128, 602, 1006, 10951, 10952, 25431, 48040, 48168}); // Inner Fire all ranks
                bool hasFortitude = hasAnyRank(p, {1243, 1244, 1245, 2791, 10937, 10938, 25389, 48161}); // PW:Fortitude all ranks
                bool hasShadowProt = hasAnyRank(p, {976, 10957, 10958, 25433, 48169}); // Shadow Protection all ranks
                bool hasDivineSpirit = hasAnyRank(p, {14752, 14818, 14819, 27841, 25312, 48073}); // Divine Spirit all ranks
                bool hasFearWard = p->HasAura(6346); // Fear Ward (single rank)
                bool shadowformActive = p->HasAura(15473); // Shadowform aura
                bool dispersionActive = p->HasAura(47585); // Dispersion aura
                bool isChanneling = p->HasUnitState(UNIT_STATE_CHANNELING);
                bool isEating = p->HasAura(433) || p->HasAura(434) || p->HasAura(435) || p->HasAura(1127) || p->HasAura(1129) || p->HasAura(1131);  // Food/Drink auras

                // Mind Blast cooldown check (1.5-8s CD depending on talents)
                bool mindBlastReady = true;
                SpellCooldowns const& cds = p->GetSpellCooldownMap();
                for (uint32 mbId : {8092u, 8102u, 8103u, 8104u, 8105u, 8106u, 10945u, 10946u, 10947u, 25372u, 25375u, 48126u, 48127u}) {
                    auto it = cds.find(mbId);
                    if (it != cds.end() && it->second.end > GameTime::GetGameTimeMS()) { mindBlastReady = false; break; }
                }
                // Psychic Scream cooldown check (30s CD)
                bool psychicScreamReady = true;
                for (uint32 psId : {8122u, 8124u, 10888u, 10890u}) {
                    auto it = cds.find(psId);
                    if (it != cds.end() && it->second.end > GameTime::GetGameTimeMS()) { psychicScreamReady = false; break; }
                }

                // Target debuffs
                bool targetHasSwPain = false;
                bool targetHasHolyFire = false;
                bool targetHasDP = false;
                bool targetHasVT = false;
                if (target) {
                    targetHasSwPain = hasAnyRank(target, {589, 594, 970, 992, 2767, 10892, 10893, 10894, 25367, 25368, 48124, 48125});
                    targetHasHolyFire = hasAnyRank(target, {14914, 15262, 15263, 15264, 15265, 15266, 15267, 15261, 25384, 48134, 48135});
                    targetHasDP = hasAnyRank(target, {2944, 19276, 19277, 19278, 19279, 19280, 25467, 48299, 48300});
                    targetHasVT = hasAnyRank(target, {34914, 34916, 34917, 48159, 48160});
                }

                ss << "\"has_shield\": \"" << (hasShield ? "true" : "false") << "\", ";
                ss << "\"target_has_sw_pain\": \"" << (targetHasSwPain ? "true" : "false") << "\", ";
                ss << "\"has_renew\": \"" << (hasRenew ? "true" : "false") << "\", ";
                ss << "\"has_inner_fire\": \"" << (hasInnerFire ? "true" : "false") << "\", ";
                ss << "\"has_fortitude\": \"" << (hasFortitude ? "true" : "false") << "\", ";
                ss << "\"mind_blast_ready\": \"" << (mindBlastReady ? "true" : "false") << "\", ";
                ss << "\"target_has_holy_fire\": \"" << (targetHasHolyFire ? "true" : "false") << "\", ";
                ss << "\"is_eating\": \"" << (isEating ? "true" : "false") << "\", ";
                ss << "\"target_has_devouring_plague\": \"" << (targetHasDP ? "true" : "false") << "\", ";
                ss << "\"has_shadow_protection\": \"" << (hasShadowProt ? "true" : "false") << "\", ";
                ss << "\"has_divine_spirit\": \"" << (hasDivineSpirit ? "true" : "false") << "\", ";
                ss << "\"has_fear_ward\": \"" << (hasFearWard ? "true" : "false") << "\", ";
                ss << "\"psychic_scream_ready\": \"" << (psychicScreamReady ? "true" : "false") << "\", ";
                ss << "\"target_has_vampiric_touch\": \"" << (targetHasVT ? "true" : "false") << "\", ";
                ss << "\"shadowform_active\": \"" << (shadowformActive ? "true" : "false") << "\", ";
                ss << "\"dispersion_active\": \"" << (dispersionActive ? "true" : "false") << "\", ";
                ss << "\"is_channeling\": \"" << (isChanneling ? "true" : "false") << "\", ";

                // Combat stats (WotLK formulas)
                float spellPower = 0;
                for (int school = 1; school < 7; ++school) {
                    float sp = p->GetUInt32Value(PLAYER_FIELD_MOD_DAMAGE_DONE_POS + school);
                    if (sp > spellPower) spellPower = sp;
                }
                float spellCrit = p->GetFloatValue(PLAYER_SPELL_CRIT_PERCENTAGE1 + 1); // Holy school
                float spellHaste = p->GetFloatValue(UNIT_MOD_CAST_SPEED) > 0 ? ((1.0f / p->GetFloatValue(UNIT_MOD_CAST_SPEED)) - 1.0f) * 100.0f : 0.0f;
                uint32 totalArmor = p->GetArmor();
                float attackPower = p->GetTotalAttackPowerValue(BASE_ATTACK);
                float meleeCrit = p->GetFloatValue(PLAYER_CRIT_PERCENTAGE);
                float dodge = p->GetFloatValue(PLAYER_DODGE_PERCENTAGE);
                float hitSpell = p->GetFloatValue(PLAYER_FIELD_COMBAT_RATING_1 + CR_HIT_SPELL) / 26.23f; // rating to %
                float expertise = p->GetFloatValue(PLAYER_EXPERTISE) * 0.25f; // expertise to dodge/parry reduction %
                float armorPen = p->GetFloatValue(PLAYER_FIELD_COMBAT_RATING_1 + CR_ARMOR_PENETRATION) / 13.99f;

                ss << "\"spell_power\": " << (int)spellPower << ", ";
                ss << "\"spell_crit\": " << spellCrit << ", ";
                ss << "\"spell_haste\": " << spellHaste << ", ";
                ss << "\"total_armor\": " << totalArmor << ", ";
                ss << "\"attack_power\": " << attackPower << ", ";
                ss << "\"melee_crit\": " << meleeCrit << ", ";
                ss << "\"dodge\": " << dodge << ", ";
                ss << "\"hit_spell\": " << hitSpell << ", ";
                ss << "\"expertise\": " << expertise << ", ";
                ss << "\"armor_pen\": " << armorPen << ", ";

                ss << "\"target_status\": \"" << tStatus << "\", ";
                ss << "\"target_hp\": " << tHp << ", ";
                ss << "\"target_level\": " << tLevel << ", ";
                ss << "\"xp_gained\": " << ev.xp_gained << ", ";
                ss << "\"loot_copper\": " << ev.loot_copper << ", ";
                ss << "\"loot_score\": " << ev.loot_score << ", ";
                ss << "\"leveled_up\": \"" << (ev.leveled_up ? "true" : "false") << "\", ";
                ss << "\"tx\": " << tx << ", ";
                ss << "\"ty\": " << ty << ", ";
                ss << "\"tz\": " << tz << ", ";
                std::string playerName = p->GetName();
                auto mobIt = _cachedNearbyMobsPerPlayer.find(playerName);
                std::string playerMobsJson = (mobIt != _cachedNearbyMobsPerPlayer.end()) ? mobIt->second : "[]";
                ss << "\"nearby_mobs\": " << playerMobsJson;
                ss << "}";
            }
            ss << "] }";
            { std::lock_guard<std::mutex> lock(g_Mutex); g_CurrentJsonState = ss.str(); g_HasNewState = true; }
            g_StateVersion.fetch_add(1, std::memory_order_relaxed);

        }

        if (_slowTimer >= 2000) {
            _slowTimer = 0;
            std::vector<Player*> players;
            CollectOnlinePlayers(players);
            std::unordered_map<std::string, std::string> newMobsPerPlayer;
            for (Player* p : players) {
                if (!p) continue;
                CreatureCollector collector(p);
                Cell::VisitObjects(p, collector, 50.0f);
                std::stringstream mobSS;
                mobSS << "[";
                bool firstMob = true;
                for (Creature* c : collector.foundCreatures) {
                    if (!firstMob) mobSS << ", ";
                    mobSS << "{";
                    mobSS << "\"guid\": \"" << c->GetGUID().GetRawValue() << "\", ";
                    mobSS << "\"name\": \"" << c->GetName() << "\", ";
                    mobSS << "\"level\": " << (int)c->GetLevel() << ", ";
                    mobSS << "\"attackable\": " << (p->IsValidAttackTarget(c) ? "1" : "0") << ", ";
                    mobSS << "\"vendor\": " << (c->IsVendor() ? "1" : "0") << ", ";
                    uint64 targetGuid = 0;
                    if (c->GetTarget()) targetGuid = c->GetTarget().GetRawValue();
                    mobSS << "\"target\": \"" << targetGuid << "\", ";
                    mobSS << "\"hp\": " << c->GetHealth() << ", ";
                    mobSS << "\"x\": " << c->GetPositionX() << ", ";
                    mobSS << "\"y\": " << c->GetPositionY() << ", ";
                    mobSS << "\"z\": " << c->GetPositionZ();
                    mobSS << "}";
                    firstMob = false;
                }
                mobSS << "]";
                newMobsPerPlayer[p->GetName()] = mobSS.str();
            }
            _cachedNearbyMobsPerPlayer = std::move(newMobsPerPlayer);
        }
    }
};

class AIControllerPlayerScript : public PlayerScript {
public:
    AIControllerPlayerScript() : PlayerScript("AIControllerPlayerScript") {}

    void OnPlayerBeforeSendChatMessage(Player* player, uint32& type, uint32& lang, std::string& msg) override {
        std::string commandPrefix = "#spawn";
        std::string commandSpawnAll = "#spawnbots";

        auto spawnBotByName = [player, &msg](std::string const& botName) -> bool {
            if (botName.empty()) {
                ChatHandler(player->GetSession()).SendSysMessage("Bot-Name fehlt.");
                return false;
            }

            if (ObjectAccessor::FindPlayerByName(botName)) {
                ChatHandler(player->GetSession()).SendSysMessage("Bot ist bereits online.");
                return false;
            }

            LOG_INFO("module", "AI-DEBUG: Versuche Bot zu spawnen: '{}'", botName);

            // ACHTUNG: Hier muss sCharacterCache evtl. durch eine entsprechende Methode in Ihrem Core ersetzt werden,
            // falls GetCharacterGuidByName dort anders heißt.
            ObjectGuid guid = sCharacterCache->GetCharacterGuidByName(botName);
            if (!guid) {
                ChatHandler(player->GetSession()).SendSysMessage("Charakter nicht gefunden.");
                return false;
            }

            LOG_INFO("module", "DEBUG STEP 1: Hole AccountID...");
            uint32 accountId = sCharacterCache->GetCharacterAccountIdByGuid(guid);
            LOG_INFO("module", "DEBUG STEP 2: AccountID ist {}", accountId);

            if (accountId == 0) {
                ChatHandler(player->GetSession()).SendSysMessage("Fehler: Ungültige AccountID.");
                return false;
            }

            if (sWorldSessionMgr->FindSession(accountId)) {
                LOG_ERROR("module", "ABORT: Account {} ist bereits eingeloggt!", accountId);
                ChatHandler(player->GetSession()).SendSysMessage("Fehler: Account bereits eingeloggt.");
                return false;
            }

            {
                std::lock_guard<std::mutex> lock(g_BotSessionsMutex);
                if (g_BotSessions.find(accountId) != g_BotSessions.end()) {
                    LOG_ERROR("module", "ABORT: Bot-Session für Account {} existiert bereits.", accountId);
                    ChatHandler(player->GetSession()).SendSysMessage("Fehler: Bot bereits aktiv.");
                    return false;
                }
            }

            // --- ASYNCHRONE LADESTRATEGIE (Core-Login-Flow) ---
            LOG_INFO("module", "DEBUG STEP 3: Starte asynchronen Login...");

            // 1. Session erstellen (IsBot simuliert durch Flags oder manuelles Handling, da Konstruktor nur 12 Args hat)
            WorldSession* botSession = new WorldSession(
                accountId,
                std::string(botName),
                0,      // Security Token (dummy)
                nullptr,// Socket (dummy)
                SEC_PLAYER,
                EXPANSION_WRATH_OF_THE_LICH_KING,
                time_t(0),
                LOCALE_enUS,
                0,
                false,
                true,   // skipQueue = true
                0
            );

            // 2. Holder erstellen
            auto holder = std::make_shared<BotLoginQueryHolder>(accountId, guid);

            // 3. Initialisieren (nur PreparedStatements setzen)
            if (!holder->Initialize()) {
                LOG_ERROR("module", "FAIL: Konnte LoginQueryHolder nicht initialisieren.");
                ChatHandler(player->GetSession()).SendSysMessage("Interner DB-Fehler (Init).");
                delete botSession;
                return false;
            }

            LOG_INFO("module", "DEBUG STEP 4: DB-Queries gestartet...");

            {
                std::lock_guard<std::mutex> lock(g_BotSessionsMutex);
                g_BotSessions[accountId] = botSession;
            }

            g_QueryHolderProcessor.AddCallback(CharacterDatabase.DelayQueryHolder(holder)).AfterComplete(
                [accountId, guid, botName](SQLQueryHolderBase const& holderBase)
                {
                    WorldSession* session = nullptr;
                    {
                        std::lock_guard<std::mutex> lock(g_BotSessionsMutex);
                        auto it = g_BotSessions.find(accountId);
                        if (it != g_BotSessions.end())
                            session = it->second;
                    }
                    if (!session)
                    {
                        LOG_ERROR("module", "Bot-Login: Session für Account {} nicht gefunden.", accountId);
                        return;
                    }

                    auto const& typedHolder = static_cast<BotLoginQueryHolder const&>(holderBase);
                    Player* botPlayer = new Player(session);

                    if (!botPlayer->LoadFromDB(guid, typedHolder))
                    {
                        LOG_ERROR("module", "FAIL: Player konnte nicht geladen werden.");
                        delete botPlayer;
                        std::lock_guard<std::mutex> lock(g_BotSessionsMutex);
                        auto it = g_BotSessions.find(accountId);
                        if (it != g_BotSessions.end())
                        {
                            delete it->second;
                            g_BotSessions.erase(it);
                        }
                        return;
                    }

                    LOG_INFO("module", "DEBUG STEP 5: Player geladen: {}", botPlayer->GetName());

                    session->SetPlayer(botPlayer);

                    botPlayer->GetMotionMaster()->Initialize();
                    botPlayer->SendInitialPacketsBeforeAddToMap();

                    ObjectAccessor::AddObject(botPlayer);

                    if (!botPlayer->GetMap()->AddPlayerToMap(botPlayer) || !botPlayer->CheckInstanceLoginValid())
                    {
                        AreaTriggerTeleport const* at = sObjectMgr->GetGoBackTrigger(botPlayer->GetMapId());
                        if (at)
                        {
                            botPlayer->TeleportTo(at->target_mapId, at->target_X, at->target_Y, at->target_Z, botPlayer->GetOrientation());
                        }
                        else
                        {
                            botPlayer->TeleportTo(botPlayer->m_homebindMapId, botPlayer->m_homebindX, botPlayer->m_homebindY, botPlayer->m_homebindZ, botPlayer->GetOrientation());
                        }

                        botPlayer->GetSession()->SendNameQueryOpcode(botPlayer->GetGUID());
                    }

                    botPlayer->SendInitialPacketsAfterAddToMap();

                    CharacterDatabasePreparedStatement* onlineStmt = CharacterDatabase.GetPreparedStatement(CHAR_UPD_CHAR_ONLINE);
                    onlineStmt->SetData(0, botPlayer->GetGUID().GetCounter());
                    CharacterDatabase.Execute(onlineStmt);

                    constexpr uint32 kSpawnMapId = 0;
                    constexpr float kSpawnX = -8921.037f;
                    constexpr float kSpawnY = -120.484985f;
                    constexpr float kSpawnZ = 82.02542f;
                    constexpr float kSpawnO = 3.299f;

                    botPlayer->TeleportTo(kSpawnMapId, kSpawnX, kSpawnY, kSpawnZ, kSpawnO);

                    // Teach priest spells if not already known
                    constexpr uint32 priestSpells[] = { 585, 2050, 589, 17 }; // Smite, Lesser Heal, SW:Pain, PW:Shield
                    for (uint32 spellId : priestSpells) {
                        if (!botPlayer->HasSpell(spellId)) {
                            botPlayer->learnSpell(spellId);
                            LOG_INFO("module", "AI-DEBUG: Bot '{}' lernt Spell {}", botName, spellId);
                        }
                    }

                    LOG_INFO("module", "DEBUG STEP 6: Bot '{}' erfolgreich gespawnt.", botName);
                });

            msg = "";
            return true;
            };

        if (msg.length() >= commandSpawnAll.length() && msg.substr(0, commandSpawnAll.length()) == commandSpawnAll) {
            LOG_INFO("module", "AI-DEBUG: Chat von {}: '{}'", player->GetName(), msg);
            std::vector<std::string> botNames = { "Bota", "Botb", "Botc", "Botd", "Bote" };
            for (std::string const& botName : botNames) {
                spawnBotByName(botName);
            }
            msg = "";
            return;
        }

        if (msg.length() >= commandPrefix.length() && msg.substr(0, commandPrefix.length()) == commandPrefix) {

            LOG_INFO("module", "AI-DEBUG: Chat von {}: '{}'", player->GetName(), msg);

            if (msg.length() <= commandPrefix.length() + 1) {
                ChatHandler(player->GetSession()).SendSysMessage("Benutzung: #spawn <BotName>");
                msg = "";
                return;
            }

            std::string botName = msg.substr(commandPrefix.length() + 1);
            if (!botName.empty() && botName.back() == ' ') botName.pop_back();

            spawnBotByName(botName);
        }
    }
    void OnPlayerGiveXP(Player* player, uint32& amount, Unit* victim, uint8 xpSource) override {
        AddXPGained(player, amount);
    }
    void OnPlayerLevelChanged(Player* player, uint8 oldLevel) override {
        if (player->GetLevel() >= 2) {
            SetLeveledUp(player);
            player->SetLevel(1); player->SetUInt32Value(PLAYER_XP, 0); player->InitStatsForLevel(true);
            player->SetHealth(player->GetMaxHealth()); player->SetPower(player->getPowerType(), player->GetMaxPower(player->getPowerType()));
        }
    }
    void OnPlayerMoneyChanged(Player* player, int32& amount) override {
        if (amount > 0) { AddLootCopper(player, amount); }
    }
};

void AddAIControllerScripts() {
    new AIControllerPlayerScript();
    new AIControllerWorldScript();
}

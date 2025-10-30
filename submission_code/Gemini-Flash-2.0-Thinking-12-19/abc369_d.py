def solve():
    n = int(input())
    a = list(map(int, input().split()))
    dp = {}

    def get_max_experience(monster_index, defeated_count):
        if monster_index == n:
            return 0
        if (monster_index, defeated_count) in dp:
            return dp[(monster_index, defeated_count)]
        
        # Option 1: Let go of the current monster
        experience_let_go = get_max_experience(monster_index + 1, defeated_count)
        
        # Option 2: Defeat the current monster
        experience_defeat = 0
        defeated_monster_index = defeated_count + 1
        experience_from_monster = a[monster_index]
        if defeated_monster_index % 2 == 0:
            experience_from_monster *= 2
        experience_defeat = experience_from_monster + get_max_experience(monster_index + 1, defeated_count + 1)
        
        result = max(experience_let_go, experience_defeat)
        dp[(monster_index, defeated_count)] = result
        return result

    max_total_experience = get_max_experience(0, 0)
    print(max_total_experience)

if __name__ == '__main__':
    solve()
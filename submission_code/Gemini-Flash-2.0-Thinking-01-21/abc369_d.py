def solve():
    n = int(input())
    a = list(map(int, input().split()))
    memo = {}
    
    def get_max_experience(monster_index, defeated_count):
        if monster_index == n:
            return 0
        if (monster_index, defeated_count) in memo:
            return memo[(monster_index, defeated_count)]
        
        # Option 1: Let monster go
        experience_let_go = get_max_experience(monster_index + 1, defeated_count)
        
        # Option 2: Defeat monster
        bonus = 0
        if (defeated_count + 1) % 2 == 0:
            bonus = a[monster_index]
        experience_defeat = a[monster_index] + bonus + get_max_experience(monster_index + 1, defeated_count + 1)
        
        result = max(experience_let_go, experience_defeat)
        memo[(monster_index, defeated_count)] = result
        return result
        
    max_total_experience = get_max_experience(0, 0)
    print(max_total_experience)

if __name__ == '__main__':
    solve()
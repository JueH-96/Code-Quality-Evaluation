def solve():
    n = int(input())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    
    mod = 998244353
    
    count = 0
    
    def is_valid(sequences):
        s_list = []
        t_list = []
        for seq in sequences:
            s_list.append(tuple(seq))
            t_list.append(tuple(reversed(seq)))
        
        combined = sorted(s_list + t_list)
        
        
        
        a_calc = [0] * n
        b_calc = [0] * n
        
        for i in range(n):
            a_calc[i] = combined.index(s_list[i]) + 1
            b_calc[i] = combined.index(t_list[i]) + 1
        
        
        for i in range(n):
            if a_calc[i] != a[i]:
                return False
            if b[i] != -1 and b_calc[i] != b[i]:
                return False
        
        return True
    
    
    def generate_sequences(index, current_sequences, used_nums):
        nonlocal count
        if index == n:
            if is_valid(current_sequences):
                count = (count + 1) % mod
            return
        
        for i1 in range(1, n + 1):
            if (index, 0, i1) in used_nums:
                continue
            for i2 in range(1, n + 1):
                if (index, 1, i2) in used_nums:
                    continue
                for i3 in range(1, n + 1):
                    if (index, 2, i3) in used_nums:
                        continue
                    
                    new_used_nums = set(used_nums)
                    new_used_nums.add((index, 0, i1))
                    new_used_nums.add((index, 1, i2))
                    new_used_nums.add((index, 2, i3))
                    
                    generate_sequences(index + 1, current_sequences + [[i1, i2, i3]], new_used_nums)
    
    generate_sequences(0, [], set())
    print(count)

solve()
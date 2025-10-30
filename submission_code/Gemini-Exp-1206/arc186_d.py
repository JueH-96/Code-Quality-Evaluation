def solve():
    n = int(input())
    a = list(map(int, input().split()))
    mod = 998244353

    def is_polish(seq):
        if len(seq) == 1 and seq[0] == 0:
            return True
        
        v1 = seq[0]
        if v1 == 0:
            return False

        
        idx = 1
        polish_counts = 0
        for _ in range(v1):
            
            sub_seq_len = 0
            stack = []
            
            while idx < len(seq):
                
                stack.append(seq[idx])
                sub_seq_len += 1
                idx += 1
                
                while stack and stack[-1] == 0:
                    stack.pop()
                    if stack:
                        stack[-1] -= 1
                
                if not stack:
                    polish_counts += 1
                    break
            
            if not stack:
                continue
            else:
                return False
        
        if polish_counts == v1 and idx == len(seq):
            return True
        else:
            return False

    def generate_polish(length, current_seq, count):
        if length == 0:
            if is_polish(current_seq):
                
                is_smaller = False
                for i in range(len(current_seq)):
                    if current_seq[i] < a[i]:
                        is_smaller = True
                        break
                    elif current_seq[i] > a[i]:
                        return count
                if is_smaller or current_seq == a:
                    return (count + 1) % mod
                else:
                    return count
            else:
                return count

        
        total_count = count
        for i in range(n):
            total_count = generate_polish(length - 1, current_seq + [i], total_count)
        return total_count
    
    
    dp = {}

    def solve_dp(idx, remaining):
        if (idx, remaining) in dp:
            return dp[(idx, remaining)]

        if idx == n:
            return 1 if remaining == 0 else 0

        ans = 0
        if remaining == 1:
            ans = solve_dp(idx + 1, 0)
        elif a[idx] > 0:
            for i in range(1, min(a[idx], n - idx) + 1):
                if remaining == 0:
                    ans = (ans + solve_dp(idx + 1, i)) % mod
                elif remaining >= i + 1:
                    ans = (ans + solve_dp(idx + 1, remaining - 1 + i)) % mod
                if i < a[idx]:
                    break
        
        dp[(idx, remaining)] = ans
        return ans

    print(solve_dp(0, 0))

solve()
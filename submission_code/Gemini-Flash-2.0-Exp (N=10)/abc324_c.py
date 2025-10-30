def solve():
    n, t_prime = input().split()
    n = int(n)
    s_list = [input() for _ in range(n)]
    
    result_indices = []
    for i, s in enumerate(s_list):
        if len(s) == len(t_prime):
            diff_count = 0
            for j in range(len(s)):
                if s[j] != t_prime[j]:
                    diff_count += 1
            if diff_count <= 1:
                result_indices.append(i + 1)
        elif len(s) == len(t_prime) - 1:
            
            found = False
            for j in range(len(t_prime)):
                temp = list(t_prime)
                temp.pop(j)
                if "".join(temp) == s:
                    found = True
                    break
            if found:
                result_indices.append(i+1)
        elif len(s) == len(t_prime) + 1:
            found = False
            for j in range(len(s)):
                temp = list(s)
                temp.pop(j)
                if "".join(temp) == t_prime:
                    found = True
                    break
            if found:
               result_indices.append(i+1)
    
    print(len(result_indices))
    if result_indices:
        print(*result_indices)

solve()
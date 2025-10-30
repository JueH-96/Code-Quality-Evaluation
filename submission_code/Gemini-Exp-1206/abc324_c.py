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
            j = 0
            k = 0
            while j < len(s) and k < len(t_prime):
                if s[j] == t_prime[k]:
                    j += 1
                    k += 1
                else:
                    k += 1
            if j == len(s):
                result_indices.append(i + 1)
        elif len(s) == len(t_prime) + 1:
            j = 0
            k = 0
            while j < len(s) and k < len(t_prime):
                if s[j] == t_prime[k]:
                    j += 1
                    k += 1
                else:
                    j += 1
            if k == len(t_prime):
                result_indices.append(i + 1)

    print(len(result_indices))
    print(*result_indices)

solve()
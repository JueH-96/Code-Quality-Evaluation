# YOUR CODE HERE
def solve():
    n, t_prime = input().split()
    n = int(n)
    s_list = [input() for _ in range(n)]
    
    ans = []
    for i, s in enumerate(s_list):
        if s == t_prime:
            ans.append(i + 1)
            continue
        
        len_s = len(s)
        len_t_prime = len(t_prime)
        
        #insertion
        if len_s + 1 == len_t_prime:
            for j in range(len_s + 1):
                temp = list(s)
                temp.insert(j, t_prime[j])
                if "".join(temp) == t_prime:
                    ans.append(i+1)
                    break

        #deletion
        if len_s -1 == len_t_prime:
            for j in range(len_s):
                temp = list(s)
                del temp[j]
                if "".join(temp) == t_prime:
                    ans.append(i+1)
                    break

        #change
        if len_s == len_t_prime:
            diff_count = 0
            for j in range(len_s):
                if s[j] != t_prime[j]:
                    diff_count +=1
            if diff_count == 1:
                ans.append(i+1)

    print(len(ans))
    print(*ans)

solve()
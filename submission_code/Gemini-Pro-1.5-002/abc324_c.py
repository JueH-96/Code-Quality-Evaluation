# YOUR CODE HERE
def solve():
    n, t_prime = input().split()
    n = int(n)
    s = [input() for _ in range(n)]
    
    result = []
    for i in range(n):
        si = s[i]
        if si == t_prime:
            result.append(i + 1)
        elif len(si) == len(t_prime) - 1:
            
            possible = False
            for j in range(len(t_prime)):
                temp = list(t_prime)
                del temp[j]
                if "".join(temp) == si:
                    possible = True
                    break
            if possible:
                result.append(i+1)
        elif len(si) == len(t_prime) + 1:
            possible = False
            for j in range(len(si)):
                temp = list(si)
                del temp[j]
                if "".join(temp) == t_prime:
                    possible = True
                    break
            if possible:
                result.append(i+1)
        elif len(si) == len(t_prime):
            diff = 0
            for j in range(len(si)):
                if si[j] != t_prime[j]:
                    diff += 1
            if diff == 1:
                result.append(i+1)

    print(len(result))
    print(*result)

solve()
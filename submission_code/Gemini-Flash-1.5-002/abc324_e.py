# YOUR CODE HERE
def solve():
    n, t = input().split()
    n = int(n)
    s = [input() for _ in range(n)]
    
    count = 0
    for i in range(n):
        for j in range(n):
            combined = s[i] + s[j]
            
            k=0
            for char in combined:
                if k < len(t) and char == t[k]:
                    k += 1
            if k == len(t):
                count += 1
    print(count)

solve()
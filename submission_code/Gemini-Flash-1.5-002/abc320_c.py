# YOUR CODE HERE
M = int(input())
S1 = input()
S2 = input()
S3 = input()

ans = -1
for i in range(M):
    c1 = S1[i]
    for j in range(M):
        c2 = S2[j]
        if c1 == c2:
            for k in range(M):
                c3 = S3[k]
                if c1 == c3:
                    t = max(i, j, k)
                    if ans == -1 or t < ans:
                        ans = t
                        
                    
                    
                    
print(ans)
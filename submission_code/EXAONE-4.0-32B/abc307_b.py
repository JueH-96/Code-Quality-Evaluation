n = int(input().strip())
words = [input().strip() for _ in range(n)]

for i in range(n):
    for j in range(n):
        if i == j:
            continue
        concat_str = words[i] + words[j]
        if concat_str == concat_str[::-1]:
            print("Yes")
            exit(0)
            
print("No")
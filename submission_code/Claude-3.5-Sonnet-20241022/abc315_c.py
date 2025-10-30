N = int(input())
flavors = []
deliciousness = []

for _ in range(N):
    f, s = map(int, input().split())
    flavors.append(f)
    deliciousness.append(s)

max_satisfaction = 0

for i in range(N):
    for j in range(i+1, N):
        s = max(deliciousness[i], deliciousness[j])
        t = min(deliciousness[i], deliciousness[j])
        
        if flavors[i] != flavors[j]:
            satisfaction = s + t
        else:
            satisfaction = s + t//2
            
        max_satisfaction = max(max_satisfaction, satisfaction)

print(max_satisfaction)
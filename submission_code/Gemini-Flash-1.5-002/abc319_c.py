import itertools

c = []
for i in range(3):
    row = list(map(int, input().split()))
    c.append(row)

count = 0
total = 0

for perm in itertools.permutations(list(itertools.chain.from_iterable(c))):
    total += 1
    disappointed = False
    seen = set()
    for i in range(9):
        seen.add(perm[i])
        for j in range(i):
            if perm[i] == perm[j]:
                
                
                lines = []
                
                #check rows
                for r in range(3):
                    lines.append(c[r])
                
                #check cols
                for col in range(3):
                    lines.append([c[row][col] for row in range(3)])
                
                #check diags
                lines.append([c[i][i] for i in range(3)])
                lines.append([c[i][2-i] for i in range(3)])
                
                
                found_line = False
                for line in lines:
                    if perm[i] in line and perm[j] in line:
                        
                        if line.index(perm[i]) < line.index(perm[j]):
                            
                            if len(set(line)) > 2:
                                disappointed = True
                                break
                        else:
                            if len(set(line)) > 2:
                                disappointed = True
                                break
                if disappointed:
                    break
    if not disappointed and len(seen) == 9:
        count += 1

print(count / total)
# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

index = 0
t = int(data[index])
index += 1
results = []

for _ in range(t):
    n = int(data[index])
    k = int(data[index + 1])
    index += 2
    a = list(map(int, data[index:index + n]))
    index += n
    
    if k == 2:
        count_even = sum(1 for x in a if x % 2 == 0)
        if count_even > 0:
            results.append(0)
        else:
            results.append(1)
    elif k == 3:
        count_div3 = sum(1 for x in a if x % 3 == 0)
        if count_div3 > 0:
            results.append(0)
        else:
            count_mod1 = sum(1 for x in a if x % 3 == 1)
            count_mod2 = sum(1 for x in a if x % 3 == 2)
            results.append(min(count_mod1, count_mod2))
    elif k == 4:
        count_div2 = sum(1 for x in a if x % 2 == 0)
        count_div4 = sum(1 for x in a if x % 4 == 0)
        if count_div4 > 0:
            results.append(0)
        elif count_div2 >= 2:
            results.append(0)
        else:
            results.append(1)
    elif k == 5:
        count_div5 = sum(1 for x in a if x % 5 == 0)
        if count_div5 > 0:
            results.append(0)
        else:
            results.append(1)

print("
".join(map(str, results)))
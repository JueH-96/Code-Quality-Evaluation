# YOUR CODE HERE
n = int(input())
days = list(map(int, input().split()))

count = 0
for i in range(1, n+1):
    for j in range(1, days[i-1]+1):
        # Convert to strings and get all unique digits
        all_digits = set(str(i) + str(j))
        
        # Check if all digits are the same (i.e., only one unique digit)
        if len(all_digits) == 1:
            count += 1

print(count)
def max_probability_same_number(dice):
    n = len(dice)
    max_prob = 0
    
    for i in range(n):
        for j in range(i+1, n):
            K_i = dice[i][0]
            nums_i = dice[i][1:K_i+1]
            
            K_j = dice[j][0]
            nums_j = dice[j][1:K_j+1]
            
            # Count occurrences of each number
            count_i = {}
            for num in nums_i:
                count_i[num] = count_i.get(num, 0) + 1
            
            count_j = {}
            for num in nums_j:
                count_j[num] = count_j.get(num, 0) + 1
            
            # Calculate probability
            prob = 0
            for num in set(count_i.keys()).intersection(set(count_j.keys())):
                prob += (count_i[num] / K_i) * (count_j[num] / K_j)
            
            max_prob = max(max_prob, prob)
    
    return max_prob

# Read input
n = int(input())
dice = []
for _ in range(n):
    die = list(map(int, input().split()))
    dice.append(die)

# Calculate and print the result
result = max_probability_same_number(dice)
print(result)
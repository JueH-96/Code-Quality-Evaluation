# YOUR CODE HERE
S = input()
result = "Yes"
for i in range(2, 17, 2):
    if S[i-1] == '1':
        result = "No"
        break
print(result)
S = input()
count = 0
for i in range(len(S)):
  for j in range(i + 1, len(S)):
    k = j + (j - i)
    if k < len(S):
      if S[i] == 'A' and S[j] == 'B' and S[k] == 'C':
        count += 1
print(count)
S = input().strip()
N = int(input())

res = []
for i in range(len(S)):
    c = S[i]
    if c == '0' or c == '1':
        res.append(c)
    else:
        # Generate candidate by setting current to 1 and remaining to 1
        candidate = ''.join(res) + '1' + '1' * (len(S) - i - 1)
        candidate_int = int(candidate, 2)
        if candidate_int <= N:
            res.append('1')
        else:
            res.append('0')

# Check the final candidate
final = ''.join(res)
final_int = int(final, 2)
print(final_int if final_int <= N else -1)
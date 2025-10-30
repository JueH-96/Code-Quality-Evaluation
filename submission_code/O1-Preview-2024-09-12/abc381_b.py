# YOUR CODE HERE
S = input()
if len(S) % 2 != 0:
    print('No')
else:
    valid = True
    for i in range(0, len(S), 2):
        if S[i] != S[i+1]:
            valid = False
            break
    if not valid:
        print('No')
    else:
        from collections import Counter
        counts = Counter(S)
        if all(count == 2 for count in counts.values()):
            print('Yes')
        else:
            print('No')
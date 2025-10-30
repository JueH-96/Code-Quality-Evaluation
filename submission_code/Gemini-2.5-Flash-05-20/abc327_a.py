N = int(input())
S = input()

found_adjacent_ab = False
for i in range(N - 1):
    char1 = S[i]
    char2 = S[i+1]

    # Check for 'a' followed by 'b' or 'b' followed by 'a'
    if (char1 == 'a' and char2 == 'b') or \
       (char1 == 'b' and char2 == 'a'):
        found_adjacent_ab = True
        break

if found_adjacent_ab:
    print("Yes")
else:
    print("No")
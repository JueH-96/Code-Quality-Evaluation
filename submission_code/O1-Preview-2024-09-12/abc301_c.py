# YOUR CODE HERE
S = input()
T = input()

valid_letters = set('atcoder')

from collections import Counter

count_S = Counter(S)
count_T = Counter(T)

ats_S = count_S.get('@', 0)
ats_T = count_T.get('@', 0)

letters = set(S+T) - set('@')

for letter in letters:
    D = count_S.get(letter, 0) - count_T.get(letter, 0)
    if letter not in valid_letters:
        if D != 0:
            print('No')
            exit()
    else:
        if D > 0:
            ats_T -= D
            if ats_T < 0:
                print('No')
                exit()
        elif D < 0:
            ats_S += D  # D is negative, subtracting negative
            if ats_S < 0:
                print('No')
                exit()
print('Yes')
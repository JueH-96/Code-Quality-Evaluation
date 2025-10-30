# YOUR CODE HERE
n = int(input())
s = input()
takahashi_wins = s.count('T')
aoki_wins = s.count('A')

if takahashi_wins > aoki_wins:
    print('T')
elif takahashi_wins < aoki_wins:
    print('A')
else:
    if s.rfind('T') > s.rfind('A'):
        print('T')
    else:
        print('A')
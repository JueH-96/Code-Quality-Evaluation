# YOUR CODE HERE
n = int(input())
s = input()
t_wins = 0
a_wins = 0
for i in range(n):
    if s[i] == 'T':
        t_wins += 1
    else:
        a_wins += 1
    if t_wins > n // 2:
        print('T')
        break
    elif a_wins > n // 2:
        print('A')
        break
else:
    if t_wins > a_wins:
        print('T')
    elif a_wins > t_wins:
        print('A')
    else:
        t_count = 0
        a_count = 0
        for char in s:
            if char == 'T':
                t_count += 1
            else:
                a_count += 1
            if t_count == t_wins:
                print('T')
                break
            if a_count == a_wins:
                print('A')
                break
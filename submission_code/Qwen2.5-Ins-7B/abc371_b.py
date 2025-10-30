# YOUR CODE HERE
n, m = map(int, input().split())
earliest_boys = [0] * n
is_taro = []

for i in range(m):
    a, b = map(int, input().split())
    a -= 1
    if b == 1 and earliest_boys[a] == 0:
        is_taro.append('Yes')
        earliest_boys[a] = i + 1
    else:
        is_taro.append('No')

for t in is_taro:
    print(t)
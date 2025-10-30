A, B = map(int, input().split())
answer = 0
if 1 <= B - A <= 99:
    answer += 1
if 1 <= A - B <= 99:
    answer += 1
# x = A
if A != B:
    answer += 1
print(answer)
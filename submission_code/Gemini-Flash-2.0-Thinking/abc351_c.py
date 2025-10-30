def solve():
    n = int(input())
    a = list(map(int, input().split()))

    sequence = []
    for i in range(n):
        sequence.append(a[i])
        while len(sequence) > 1:
            rightmost = sequence[-1]
            second_rightmost = sequence[-2]
            if rightmost == second_rightmost:
                sequence.pop()
                sequence.pop()
                sequence.append(rightmost + 1)
            else:
                break
    print(len(sequence))

solve()
def solve():
    n = int(input())
    a_list = list(map(int, input().split()))
    sequence_exponents = []
    for exponent in a_list:
        sequence_exponents.append(exponent)
        while len(sequence_exponents) >= 2:
            last_exponent = sequence_exponents[-1]
            second_last_exponent = sequence_exponents[-2]
            if last_exponent == second_last_exponent:
                sequence_exponents.pop()
                sequence_exponents.pop()
                sequence_exponents.append(last_exponent + 1)
            else:
                break
    print(len(sequence_exponents))

if __name__ == '__main__':
    solve()
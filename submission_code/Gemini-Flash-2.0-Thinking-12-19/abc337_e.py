import math

def solve():
    n = int(input())
    m = math.ceil(math.log2(n)) if n > 1 else 0
    if n == 1:
        print(0)
        s = ""
        print(1)
        return

    print(m)
    friend_bottles = [[] for _ in range(m)]
    for bottle_num in range(1, n + 1):
        binary_representation = bin(bottle_num - 1)[2:].zfill(m)
        for i in range(m):
            if binary_representation[m - 1 - i] == '1':
                friend_bottles[i].append(bottle_num)

    for i in range(m):
        bottles = friend_bottles[i]
        print(len(bottles), *bottles)

    s = input()
    spoiled_binary_reversed = ""
    for char in s:
        spoiled_binary_reversed += char

    spoiled_binary = spoiled_binary_reversed[::-1]
    spoiled_bottle_index = int(spoiled_binary, 2)
    spoiled_bottle_number = spoiled_bottle_index + 1
    print(spoiled_bottle_number)

if __name__ == '__main__':
    solve()
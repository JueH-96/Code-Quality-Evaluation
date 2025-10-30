import math

def solve():
    n = int(input())
    m = math.ceil(math.log2(n)) if n > 1 else 0
    if n == 1:
        print(0)
        s = input()
        print(1)
        return

    print(int(m))
    friends_bottles = []
    for i in range(int(m)):
        bottles_for_friend = []
        for bottle_num in range(1, n + 1):
            binary_representation = bin(bottle_num - 1)[2:].zfill(int(m))
            if binary_representation[int(m) - 1 - i] == '1':
                bottles_for_friend.append(bottle_num)
        friends_bottles.append(bottles_for_friend)
        print(len(bottles_for_friend), *sorted(bottles_for_friend))

    s = input()
    spoiled_bottle_index = 0
    for i in range(int(m)):
        if s[i] == '1':
            spoiled_bottle_index += (1 << (int(m) - 1 - i))
    print(spoiled_bottle_index + 1)

if __name__ == '__main__':
    solve()
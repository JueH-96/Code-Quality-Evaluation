import sys

def main():
    sys.setrecursionlimit(1 << 25)
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))

    if n == 0:
        print(0)
        return

    current_dp = set()
    first = a[0]
    initial_xor = first
    initial_available = frozenset([first])
    current_dp.add((initial_xor, initial_available))

    for num in a[1:]:
        new_dp = set()
        for (current_xor, available) in current_dp:
            # Option 1: create new subset
            new_xor = current_xor ^ num
            new_available_set = set(available)
            new_available_set.add(num)
            new_frozen = frozenset(new_available_set)
            new_dp.add((new_xor, new_frozen))

            # Option 2: add to any existing subset
            for s in available:
                new_sum = s + num
                new_xor2 = (current_xor ^ s) ^ new_sum
                new_available2 = set(available)
                new_available2.remove(s)
                new_available2.add(new_sum)
                new_frozen2 = frozenset(new_available2)
                new_dp.add((new_xor2, new_frozen2))
        current_dp = new_dp

    result = set()
    for (x, _) in current_dp:
        result.add(x)
    print(len(result))

if __name__ == "__main__":
    main()
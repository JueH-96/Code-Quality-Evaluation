from itertools import combinations

def find_kth_smallest_321_like_number(k):
    def generate_321_like_numbers():
        for length in range(1, 11):
            for combination in combinations(range(10), length):
                yield int(''.join(map(str, reversed(combination))))

    return sorted(generate_321_like_numbers())[k - 1]

k = int(input())
print(find_kth_smallest_321_like_number(k))
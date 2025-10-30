def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    
    N = int(input_data[0])
    Ks = list(map(int, input_data[1:]))

    total_people = sum(Ks)
    
    # We will iterate over all possible ways to assign departments (2^N possibilities).
    # For each subset, let subset_sum be the sum of its elements (Group A).
    # Then the other group (Group B) has sum = total_people - subset_sum.
    # We track the minimum among the maximum(subset_sum, total_people - subset_sum).
    
    min_max_lunch = float('inf')
    
    # Since N <= 20, 2^N <= 1,048,576, which is feasible to check exhaustively.
    from itertools import combinations
    
    # Instead of directly flipping bits, we can use combinations to generate subset sums more efficiently
    # but a direct bit iteration is also fine. We'll do a direct bit iteration:
    for mask in range(1 << N):
        subset_sum = 0
        for i in range(N):
            if mask & (1 << i):
                subset_sum += Ks[i]
        # The other group is total_people - subset_sum
        max_lunch = max(subset_sum, total_people - subset_sum)
        if max_lunch < min_max_lunch:
            min_max_lunch = max_lunch

    print(min_max_lunch)

# Do not forget to call main() at the end
if __name__ == "__main__":
    main()
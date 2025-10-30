import sys
import math

def solve():
    # Read N, D, P from the first line of input.
    # N: number of days of the trip
    # D: number of one-day passes in a batch
    # P: cost of one batch of D passes
    line1 = sys.stdin.readline().split()
    N = int(line1[0])
    D = int(line1[1])
    P = int(line1[2])

    # Read the list of regular fares F for N days from the second line of input.
    F = list(map(int, sys.stdin.readline().split()))

    # To minimize the total cost, we should use one-day passes for the days
    # with the highest regular fares. So, we sort the fares in descending order.
    F_sorted = sorted(F, reverse=True)

    # Compute suffix sums of the sorted fares.
    # S[i] will store the sum of F_sorted[i] + F_sorted[i+1] + ... + F_sorted[N-1].
    # This allows efficient calculation of the total regular fare for days not covered by passes.
    # S is 0-indexed based on the indices of F_sorted.
    # S[N] represents the sum of an empty suffix, which is 0.
    S = [0] * (N + 1)
    for i in range(N - 1, -1, -1):
        S[i] = F_sorted[i] + S[i + 1]

    # We need to consider how many batches of passes to buy. Let's say we buy k batches.
    # If we buy k batches, the cost of buying passes is k * P.
    # We receive k * D one-day passes.
    # We should use these passes on the min(k * D, N) days with the highest fares
    # (the first min(k * D, N) elements in F_sorted).
    # The remaining days must be paid using their regular fares.

    # Calculate the minimum number of batches required to potentially cover all N days with passes.
    # This is the smallest integer k such that k * D >= N, which is ceil(N / D).
    # Using integer division, ceil(N / D) is (N + D - 1) // D.
    k_to_cover_all = (N + D - 1) // D

    # One potential minimum total cost is to cover all N days using passes.
    # This requires buying k_to_cover_all batches, costing k_to_cover_all * P.
    # If we cover all days, the cost of regular fares is 0.
    min_total_cost = k_to_cover_all * P

    # We also need to consider options where we buy k batches, where k < k_to_cover_all.
    # The number of batches k can be 0, 1, 2, ..., up to k_to_cover_all - 1.
    # For each such k, we buy k batches, costing k * P.
    # We have k * D passes. Since k < k_to_cover_all, k * D is less than N
    # (specifically, k * D < N based on the math derived in thought process).
    # We use these k * D passes on the first k * D days in F_sorted.
    # We pay regular fares for the remaining N - (k * D) days.
    # These remaining days correspond to the fares from index k * D onwards in F_sorted.
    # The sum of fares for these remaining days is S[k * D].
    for k in range(k_to_cover_all):
        # Number of days covered by passes if we buy k batches
        days_covered_count = k * D

        # Cost of buying k batches of passes
        cost_passes = k * P

        # Cost of regular fares for the remaining days.
        # This is the sum of fares from index days_covered_count to N-1 in F_sorted, which is S[days_covered_count].
        # Note: days_covered_count = k * D is guaranteed to be < N for k < k_to_cover_all,
        # so S[days_covered_count] is a valid index into S[0...N].
        cost_regular = S[days_covered_count]

        # The total cost for buying k batches is the cost of passes plus the cost of regular fares for uncovered days.
        current_total_cost = cost_passes + cost_regular

        # Update the minimum total cost found so far.
        min_total_cost = min(min_total_cost, current_total_cost)

    # The minimum_total_cost now holds the minimum cost found by considering
    # all relevant options for the number of batches (from 0 up to k_to_cover_all, implicitly).
    # Print the minimum total cost.
    print(min_total_cost)

solve()
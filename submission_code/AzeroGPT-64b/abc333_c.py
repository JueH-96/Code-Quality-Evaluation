def generate_repunits(n):
    # Generate n repunits starting from 11, 111, 1111... and so on.
    repunits = ['1']
    for _ in range(1, n):
        repunits.append(str(int(repunits[-1]) + int(repunits[-1][-1] + "0" * (len(repunits[-1]) - 1))))
    return repunits

def nth_sum_three_repunits(rank):
    n, candidates = 1, []
    repunits = generate_repunits(19)
    while True:
        for j in range(n, 19):
            for k in range(j, 19):
                s = int(repunits[n-1]) + int(repunits[j]) + int(repunits[k])
                candidates.append(s)
        candidates.sort()
        if len(candidates) >= rank:
            return candidates[rank - 1]
        n += 1

rank = int(input())
print(nth_sum_three_repunits(rank))
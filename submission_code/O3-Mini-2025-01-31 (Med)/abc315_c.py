def main():
    import sys
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    
    # For each flavor, store the top two deliciousness values:
    # We'll keep (max1, max2) where max1 is highest and max2 is second highest.
    flavor_best = {}
    for _ in range(N):
        f = int(next(it))
        s = int(next(it))
        if f in flavor_best:
            best, second = flavor_best[f]
            if s > best:
                flavor_best[f] = (s, best)
            elif s > second:
                flavor_best[f] = (best, s)
        else:
            flavor_best[f] = (s, 0)
    
    # Option 1: Choose two cups of different flavors:
    # The best candidate from each flavor is available, so we take the two highest among them.
    best_values = []
    for value, _ in flavor_best.values():
        best_values.append(value)
    best_values.sort(reverse=True)
    
    diff_candidate = 0
    if len(best_values) >= 2:
        diff_candidate = best_values[0] + best_values[1]
    
    # Option 2: Choose two cups from the same flavor:
    # For each flavor with at least two cups, satisfaction = s + (t // 2),
    # where s is the highest deliciousness and t is the second highest.
    same_candidate = 0
    for best, second in flavor_best.values():
        if second > 0:
            cand = best + second // 2
            if cand > same_candidate:
                same_candidate = cand
                
    ans = max(diff_candidate, same_candidate)
    sys.stdout.write(str(ans))
    
if __name__ == "__main__":
    main()
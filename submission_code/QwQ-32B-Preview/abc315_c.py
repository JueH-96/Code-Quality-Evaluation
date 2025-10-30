import sys
from collections import defaultdict

def main():
    import sys
    from collections import defaultdict
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    F = list(map(int, data[1::2]))
    S = list(map(int, data[2::2]))
    
    # Group S by F, keep top two
    flavor_to_S = defaultdict(list)
    for f, s in zip(F, S):
        flavor_to_S[f].append(s)
        flavor_to_S[f].sort(reverse=True)
        if len(flavor_to_S[f]) > 2:
            flavor_to_S[f] = flavor_to_S[f][:2]
    
    # Find max1 and max2 from different flavors
    cups = list(zip(S, F))
    cups.sort(reverse=True)
    
    max1 = -1
    max2 = -1
    flavor1 = -1
    for s, f in cups:
        if f != flavor1:
            if s > max1:
                max1 = s
                flavor1 = f
                # Find max2 with different flavor
                max2 = -1
                for s2, f2 in cups:
                    if f2 != flavor1 and s2 > max2:
                        max2 = s2
                        break
            elif s > max2:
                max2 = s
            break  # No need to continue once max1 and max2 are set
    
    satisfaction_diff = max1 + max2
    
    # Find max satisfaction for same flavors
    max_same = -1
    for flavor in flavor_to_S:
        if len(flavor_to_S[flavor]) >= 2:
            s = flavor_to_S[flavor][0]
            t = flavor_to_S[flavor][1]
            sat = s + (t // 2)
            if sat > max_same:
                max_same = sat
    
    # Final answer is the maximum of the two cases
    answer = max(satisfaction_diff, max_same)
    print(answer)

if __name__ == "__main__":
    main()
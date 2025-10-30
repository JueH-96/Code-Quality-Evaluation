def main():
    import sys
    data = sys.stdin.read().splitlines()
    n = int(data[0].strip())
    index = 1
    persons = []
    for i in range(n):
        c_i = int(data[index].strip())
        index += 1
        bets = list(map(int, data[index].split()))
        index += 1
        persons.append((c_i, bets))
    X = int(data[index].strip())
    
    candidates = []
    for i in range(n):
        c_i, bets_list = persons[i]
        if X in bets_list:
            candidates.append((c_i, i+1))
            
    if len(candidates) == 0:
        print(0)
    else:
        min_bets = min(c[0] for c in candidates)
        result_people = [i for (bets, i) in candidates if bets == min_bets]
        result_people.sort()
        print(len(result_people))
        print(" ".join(map(str, result_people)))

if __name__ == "__main__":
    main()
def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    it = iter(input_data)
    n = int(next(it))

    # For each flavor, we store up to two cups (the ones with the highest deliciousness values).
    # We need two because when choosing cups with the same flavor, the satisfaction is:
    #   s + (t // 2)
    # where s is the higher deliciousness.
    top_two_by_flavor = {}
    for _ in range(n):
        flavor = int(next(it))
        score = int(next(it))
        if flavor not in top_two_by_flavor:
            top_two_by_flavor[flavor] = [score]
        else:
            # Insert the score such that the list is kept in descending order (only two elements)
            current = top_two_by_flavor[flavor]
            # If this score is greater than the current maximum, update accordingly.
            if score > current[0]:
                if len(current) == 1:
                    top_two_by_flavor[flavor] = [score, current[0]]
                else:
                    top_two_by_flavor[flavor][1] = current[0]
                    top_two_by_flavor[flavor][0] = score
            elif len(current) == 1 or score > current[1]:
                if len(current) == 1:
                    top_two_by_flavor[flavor].append(score)
                else:
                    top_two_by_flavor[flavor][1] = score

    # Candidate 1: use two cups from the same flavor if available.
    best_satisfaction = 0
    # Also, for candidate 2 we need the best cup from each flavor.
    best_cups = []
    for flavor, scores in top_two_by_flavor.items():
        best_cups.append(scores[0])  # best score per flavor
        if len(scores) >= 2:
            # Satisfaction for choosing two cups of the same flavor is:
            # top score + (second score / 2)
            candidate = scores[0] + (scores[1] // 2)
            if candidate > best_satisfaction:
                best_satisfaction = candidate

    # Candidate 2: choose cups from two different flavors.
    # The best combination is the sum of the two highest scores among best cups from different flavors.
    if len(best_cups) >= 2:
        best_cups.sort(reverse=True)
        candidate = best_cups[0] + best_cups[1]
        if candidate > best_satisfaction:
            best_satisfaction = candidate

    print(best_satisfaction)

if __name__ == '__main__':
    main()
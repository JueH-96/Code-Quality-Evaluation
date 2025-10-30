def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    t = int(data[0])
    pos_in_data = 1
    results = []
    # Explanation:
    # We are given 2N numbers and every couple (number) appears exactly twice.
    # We number positions 1...2N.
    # The problem asks for counting pairs (a, b) (with a < b) such that:
    #   (1) The two occurrences of a are not adjacent and similarly for b.
    #   (2) By swapping an occurrence of a with an occurrence of b (allowed operation
    #       swaps an a and a b) repeatedly, it’s possible to group each couple (i.e.
    #       have the two a’s in consecutive seats and likewise for b).
    #
    # If we “focus” only on the four impacted seats (the two seats holding a and the two
    # seats holding b), the allowed operation lets us exchange the labels arbitrarily (since
    # any swap involves one a and one b). Thus, starting from the initial state (which must not
    # already have a couple adjacent by condition) the question is: can we “relabel” these four
    # fixed positions so that the two positions given the same label become consecutive in the final 
    # entire seating? Note that we are not reordering seats; we are reassigning the labels among 
    # these four positions.
    #
    # To have a couple seated consecutively in the final arrangement the two positions that get 
    # that label must be consecutive integers. Now, let a’s positions be (l_a, r_a) and b’s positions 
    # be (l_b, r_b) (with l < r in each pair). Sort these four positions into s1 < s2 < s3 < s4. 
    # For it to be possible to “group” both couples simultaneously the only possibility is to assign 
    # one couple’s label to s1 and s2 (which must then be consecutive integers, i.e. s2 = s1+1) 
    # and the other couple’s label to s3 and s4 (so s4 = s3+1). (Other assignments would fail 
    # to have both pairs consist of consecutive seat numbers.)
    #
    # A check reveals that if the two couples (say, a and b) have their occurrences interwoven in either:
    #      a, b, a, b   or   b, a, b, a 
    # then after sorting the positions one finds 
    #      s1 = min(l_a, l_b), s2 = max(l_a,l_b), s3 = min(r_a,r_b), s4 = max(r_a,r_b).
    # In order to have s2 = s1+1 and s4 = s3+1 it is necessary (and sufficient)
    # that the couple with the earlier left occurrence has its left occurrence exactly one less than
    # the other’s left occurrence, and similarly its right occurrence is exactly one less than the other’s.
    #
    # Thus, if we compute for every couple (number) its two positions (l, r), and only consider those couples
    # that are “swappable” (i.e. not already sitting adjacent, so r != l+1),
    # then for two couples with positions (l1, r1) and (l2, r2) where l1 < l2, a necessary and sufficient 
    # condition is:
    #        l2 == l1 + 1  and  r2 == r1 + 1.
    #
    # Also note that if we sort the couples by their left occurrence then any potential pair that can satisfy
    # these conditions must appear adjacent in this sorted order.
    #
    # So the plan is:
    #   1. For each number from 1 to N, record its two positions (1-indexed).
    #   2. Discard a couple if its two occurrences are adjacent.
    #   3. Sort the remaining couples by their left position.
    #   4. For each consecutive pair in this sorted list, if the second’s left equals the first’s left+1
    #      and its right equals the first’s right+1, count this pair.
    #
    # This count is the answer.
    
    # Process each test case.
    cur = pos_in_data
    # We know T test cases follow.
    out_lines = []
    for _ in range(t):
        n = int(data[cur]); cur += 1
        a_list = list(map(int, data[cur:cur + 2*n])); cur += 2*n
        # For each couple (1..n), record its left and right occurrence.
        # pos[i] = [first_occurrence, second_occurrence].
        pos = [[0, 0] for _ in range(n+1)]
        for idx, val in enumerate(a_list, start=1):
            if pos[val][0] == 0:
                pos[val][0] = idx
            else:
                pos[val][1] = idx
        valid_couples = []  # will hold (l, r) for each couple that is not already consecutive.
        for i in range(1, n+1):
            l, r = pos[i]
            if r != l + 1:  # not sitting adjacent
                valid_couples.append((l, r))
        valid_couples.sort(key=lambda pair: pair[0])
    
        count = 0
        # Only consecutive couples in valid_couples (in order of left occurrence)
        # can possibly have left2 = left1+1 and right2 = right1+1.
        for i in range(len(valid_couples)-1):
            l1, r1 = valid_couples[i]
            l2, r2 = valid_couples[i+1]
            if l2 == l1 + 1 and r2 == r1 + 1:
                count += 1
        out_lines.append(str(count))
    
    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()
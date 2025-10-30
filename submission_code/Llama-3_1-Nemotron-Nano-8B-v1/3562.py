<think>
Okay, I need to solve this problem where I have to select up to 4 non-overlapping intervals to maximize the total weight, and if there are multiple such selections, pick the lex smallest one. Hmm, let's think about how to approach this.

First, the problem says the intervals can't overlap. So, the key is to select intervals that don't overlap and have the maximum sum of weights. But since we can choose up to 4, maybe even fewer if that's better, but the examples show that sometimes 2 or 4 are chosen. Wait, the first example picks two intervals, which gives the maximum sum. So the goal is to find the best possible sum with up to 4 intervals, and then among all possible selections that achieve that sum, pick the lex smallest array of indices.

But how do I model this? It feels like a dynamic programming problem. Because for each interval, the decision to include it depends on the previous intervals that don't overlap with it.

Wait, but the intervals are not sorted. So maybe the first step is to sort them in a way that makes it easier to process. Like, sorted by their end points. Because when you select intervals, you want to pick ones that end earliest to leave more space for others. But since the weight is a factor, it's a trade-off between the weight and the positioning.

Alternatively, maybe we can model this as a weighted interval scheduling problem with a maximum of 4 intervals. The standard weighted interval scheduling problem for maximum weight with one interval is straightforward, but here we can have up to four, and we need to track the best possible combinations.

So, dynamic programming where the state is the number of intervals selected so far (up to 4) and the last end time. For each step, we can choose to take the next interval (if it doesn't overlap with the previous one) and add its weight, or skip it.

But the problem is that the intervals can be in any order, so we need to process them in a sorted order. Let's think: if we sort the intervals by their end times, then for each interval, we can find the latest interval that doesn't conflict with it. That's similar to the interval scheduling problem.

But since we can select up to 4 intervals, the DP state needs to track how many intervals have been selected and the latest end time. So, for each interval i, and for each possible count k (from 0 to 4), we can track the maximum weight achievable by selecting k intervals up to i, with the last interval ending at a certain time.

Wait, but the DP approach for the maximum weight with k intervals would need to track the last end time. So the state could be (k, last_end), where k is the number of intervals selected, and last_end is the end time of the last interval. For each interval, we can decide to include it or not. If we include it, then we can only add it if its start is after last_end, and then we update the state to (k+1, new_end). The weight would be the sum of the previous weight plus the current interval's weight.

But how do we track the lex smallest array of indices when there are multiple possibilities with the same maximum weight? That complicates things because we need to not only track the maximum weight but also the lex smallest path.

Hmm. So the DP needs to track not just the maximum weight for a state (k, last_end), but also the list of indices that form that selection, so that when multiple options have the same weight, we can choose the lex smallest one.

But storing the list of indices for each state is going to be memory-intensive, especially since the number of intervals can be up to 5e4. So that's not feasible.

Alternative approach: when building the DP, for each state, we track the maximum weight and the best (lex smallest) list of indices that achieves that weight. But even that might be too much for 5e4 intervals and 4 steps.

Wait, but perhaps the intervals can be sorted in a way that allows us to process them in order, and for each step, we can keep track of the best possible options for each k (number of intervals selected) and the earliest end time. Because for a given k, the earliest possible end time would allow more intervals to be added later, which could lead to a higher weight. But since we need the maximum weight, perhaps the DP state should track for each k, the maximum weight possible and the earliest end time for that weight, along with the lex smallest indices.

Alternatively, for each k (from 0 to 4), we can track a list of possible states, each consisting of the end time and the list of indices. But again, this could be too slow.

Wait, but the maximum k is 4, which is small. So maybe for each k, we can track the best possible state. Let's think:

For each interval in the sorted list, and for each possible k (0 to 4), we can track the best possible way to select k intervals up to that point, which gives the maximum weight. If there are multiple ways to get the same weight, we choose the lex smallest indices.

So the steps could be:

1. Sort the intervals by their end time. Then, for each interval, we can find the latest interval that doesn't overlap with it. This can be done using binary search.

2. For each interval i, and for each possible k (from 1 to 4), compute the maximum weight by considering whether to include this interval. If we include it, then we can look back to the latest interval that ends before the start of i, and take the best value for k-1.

But how to track the indices and ensure lex order?

Alternatively, for each k, we can track the best possible state, which includes the maximum weight, the earliest end time (to allow more intervals), and the lex smallest indices.

Wait, but the lex order is determined by the order of the indices. So when two selections have the same weight, the one with the lex smaller array (earlier indices) is better. So when building the DP, for each state, if two options have the same weight, we need to choose the one with the lex smaller indices.

But how to model this in the DP.

Another idea: when processing intervals in order, for each k, we can keep track of the best possible state (max weight, earliest end, and the lex smallest indices). For each new interval, we can check if adding it can improve the state for k+1.

But the problem is that the intervals are sorted by end time. So when considering interval i, all previous intervals end before or at the same time as i's start. Wait, no. If the intervals are sorted by end time, then for interval i, any interval j < i has end <= i's end. So if i's start is after j's end, then j and i are non-overlapping.

So for each interval i, sorted by end time, we can find the latest interval j where j's end <= i's start. Then, for each k, the maximum weight for selecting k intervals up to i is the max between the existing value for k intervals up to i-1, or (the value for k-1 intervals up to j) + weight of i.

But how to track the indices and the lex order.

Alternatively, for each k (0 to 4), we can track a list of possible candidates. Each candidate has:

- end time: the end of the last interval in the selection.

- total weight: the sum of the weights.

- indices: the list of indices in the selection.

For each new interval, we can check for each k, if adding this interval to a candidate in k-1 (with end <= current interval's start) would create a better candidate for k.

But with 5e4 intervals and 4 steps, this could be manageable if the number of candidates per k is kept small.

But how to manage the candidates efficiently.

Wait, perhaps for each k, we can keep track of the best possible candidate. The best candidate is the one with the maximum weight. If there are multiple candidates with the same weight, we choose the one with the earliest end time (so that future intervals can be added), and if there are still ties, the lex smallest indices.

So for each k, we can have a single best candidate. Then, when processing a new interval, for each k from 4 down to 1, we can check if adding this interval to the best candidate of k-1 (if possible) would give a better candidate for k.

This way, for each interval, we process each k in reverse order (to prevent overwriting the current best before it's used for higher k's), and update the best candidates.

But how to handle the lex order when two candidates have the same weight and end time.

For example, suppose for k=2, there are two candidates with the same weight and end time, but different indices. We need to choose the one with the lex smallest indices.

So, when comparing two candidates for the same k, the criteria are:

1. Higher weight is better.

2. If weights are equal, earlier end time is better.

3. If both weight and end time are equal, lex smaller indices array is better.

So, for each k, the best candidate is the one that is optimal according to these criteria.

So, the plan is:

1. Sort the intervals by their end time. Also, keep track of their original indices.

2. For each interval in the sorted list, process them in order.

3. For each interval, check for each k from 4 down to 1:

   a. Find the latest interval j (using binary search) that ends before the current interval's start. The best candidate for k-1 up to j is the one that can be combined with the current interval.

   b. The new candidate for k would be the candidate for k-1 (if exists) plus the current interval. The weight is the sum, the end is the current interval's end, and the indices are the previous indices plus the current index.

   c. Compare this new candidate with the existing best candidate for k. If the new candidate has a higher weight, it replaces the existing. If the weight is the same, check the end time. If the end time is earlier, it's better. If end time is same, compare the indices lex order.

4. After processing all intervals, check all k from 0 to 4, find the maximum weight. Then, among all k (up to 4) that have this maximum weight, select the one with the lex smallest indices.

Wait, but the problem says "up to 4" intervals. So the maximum could be achieved with 2 intervals, like in example 1. So after processing all intervals, we need to check all possible k (0 to 4) and find the maximum weight. Then, among all k that have that maximum weight, select the best candidate (lex smallest indices) for that k.

But how to track the best candidates for each k.

So, the steps would be:

- Sort the intervals by end time, keeping their original indices.

- Precompute for each interval i, the latest interval j where j's end <= i's start. This can be done with binary search.

- Initialize a list of best candidates for each k (0 to 4). For k=0, the best candidate is weight 0, end -infinity (or 0), and empty indices. For other k, initially, they are invalid (like, for k=1, the best candidate is None until we find a valid one).

- For each interval in the sorted list (processed in order of increasing end time):

   For k in 4 down to 1:

      Find the latest j where j's end <= current interval's start.

      If j exists (i.e., there is a best candidate for k-1 up to j), then:

          new_weight = best_candidate[k-1].weight + current interval's weight

          new_end = current interval's end

          new_indices = best_candidate[k-1].indices + [current index]

          Now, compare this new candidate with the current best_candidate[k].

          If the new candidate's weight is higher than the current best, replace it.

          If the weight is equal, check the end time. If new_end is earlier, replace.

          If weight and end are same, compare the indices lex order. If new_indices is lex smaller, replace.

   Also, for each interval, we can consider not including it. But since we process in order, the best candidates for each k are maintained, and the current interval is only considered for inclusion.

Wait, but the best_candidate[k] is the best possible up to the current interval. So, for each interval, when processing k from 4 down to 1, we check if adding this interval to the best_candidate[k-1] (from before processing this interval) would create a better candidate for k.

This way, the current interval is considered for inclusion, but the previous best_candidate[k-1] is from the intervals before the current one. Because we process k in reverse order, the current interval's inclusion for k=4 is based on the best_candidate[3] from previous steps, not the current one.

So, the algorithm would proceed as follows:

Initialize for each k (0-4) a best_candidate structure. For k=0, it's (weight=0, end=-infinity, indices=[]). For others, initially, they are (weight=-infinity, end=+infinity, indices=[]).

Then, for each interval in the sorted list:

   For k from 4 down to 1:

      Find j, the latest interval that ends before current interval's start. This j is the index in the sorted list where the end is <= current's start. So, using binary search on the sorted list.

      If j is found (i.e., there exists a best_candidate for k-1 up to j), then:

          new_weight = best_candidate_prev[k-1].weight + current weight

          new_end = current interval's end

          new_indices = best_candidate_prev[k-1].indices + [current index]

          Now, compare this new candidate with the current best_candidate[k].

          If new_weight > best_candidate[k].weight: replace.

          Else if new_weight == best_candidate[k].weight:

              if new_end < best_candidate[k].end: replace.

              else if new_end == best_candidate[k].end:

                  if new_indices is lex smaller than best_candidate[k].indices: replace.

   Update the best_candidate[k] for the current interval.

Wait, but how do we track the best_candidate_prev for k-1? Because the best_candidate[k-1] is the best up to the previous intervals. So, for each interval, when processing k in reverse order, the best_candidate[k-1] is the state before considering the current interval. So, for each interval, when processing k=4, we look at the best_candidate[3] from the previous step (before this interval was processed), then for k=3, we look at best_candidate[2] from previous step, etc.

So, during processing each interval, for each k in 4 down to 1, we check the best_candidate[k-1] from the previous state (before this interval was processed) and see if adding the current interval would create a better candidate for k.

This way, the current interval is considered for inclusion, but the previous best_candidate[k-1] is from the intervals processed before this one.

So, the key is that for each interval, when processing k in reverse order, the best_candidate[k-1] is the state before this interval was processed, so that when we process k=4, the best_candidate[3] is from the previous intervals, not including the current one.

This approach ensures that each interval is considered once, and the best candidates are updated accordingly.

Now, the problem is how to represent the best_candidate structure and efficiently compute the latest j for each interval.

First, let's sort the intervals by their end time, and also keep track of their original indices.

So, sorted_intervals = sorted(intervals, key=lambda x: x[1])

But each element in sorted_intervals is [l, r, weight, original_index]. Wait, the original input is intervals[i] = [l_i, r_i, weight_i]. So, when we sort, we need to keep track of the original index. So, perhaps we can create a list of tuples (l, r, weight, idx) and sort by r.

So:

sorted_intervals = sorted( ( (l, r, w, i) for i, (l, r, w) in enumerate(intervals) ), key=lambda x: x[1] )

Then, for each interval in sorted_intervals, we can process them in order.

For each interval in sorted_intervals, we can precompute the latest j where the end of j is <= start of current interval. Since the intervals are sorted by end, we can perform a binary search on the list up to the current index minus one.

For example, for the current interval at index i in sorted_intervals, the start is l_i. We need to find the largest j < i where sorted_intervals[j][1] <= l_i.

This can be done with binary search. The list is sorted by end, so the end of each interval is in non-decreasing order. So, the binary search can find the rightmost j where end <= l_i.

Once we have that j, we can look up the best_candidate for k-1 up to j.

But how to track the best_candidate for each j? Because the best_candidate is maintained as we process each interval. So, for each interval in the sorted list, the best_candidate for each k is the best up to that interval.

Wait, but the best_candidate is built incrementally. For example, when processing the first interval, the best_candidate for k=1 would be that interval. Then, when processing the second interval, we check if adding it to the best_candidate for k=0 (which is the initial state) would create a better candidate for k=1, etc.

But the problem is that for each interval, when we process it, the best_candidate for each k is updated. So, the best_candidate for k is the best possible up to and including the current interval.

But when we process the next interval, the best_candidate for k-1 is the one from the previous steps, not including the current interval.

So, during the processing of each interval, for each k in 4 down to 1, we need to check the best_candidate[k-1] from before processing this interval. Then, if adding the current interval to that gives a better candidate for k, we update the best_candidate[k].

This suggests that for each interval, we need to have a copy of the best_candidate array before processing the interval, so that when we process k in reverse order, the previous state is available.

Alternatively, since we process k in reverse order (from 4 to 1), and for each k, the best_candidate[k-1] is from the previous state (before this interval was processed), we can proceed as follows:

For each interval in sorted_intervals:

   create a copy of the current best_candidate array (before processing this interval)

   for k in 4 down to 1:

      j = find the latest interval in sorted_intervals (before current) where end <= current's start.

      if j is found (i.e., there is a best_candidate_prev[k-1] that is valid):

          new_candidate = best_candidate_prev[k-1] + current interval's info

          compare with current best_candidate[k] (from the copy) and update if new_candidate is better.

   update the best_candidate array with the new candidates.

But making a copy of the best_candidate array for each interval is O(1) since there are only 5 elements (k=0 to 4). So this is manageable.

So, the steps in code would be:

1. Sort the intervals by end time, keeping their original indices.

2. Precompute for each interval the latest j where end <= start of current interval.

3. Initialize the best_candidate array. For each k in 0-4, best_candidate[k] is a tuple (max_weight, end_time, indices). Initially, for k=0, it's (0, -infinity, []). For others, ( -infinity, +infinity, [] )

4. For each interval in sorted_intervals:

   a. Make a copy of the current best_candidate array (prev_best).

   b. For k in 4 down to 1:

      i. Find j using binary search on the sorted_intervals up to the current index -1.

      ii. If j is found (i.e., there is a best_candidate_prev[k-1] with max_weight >=0):

          new_weight = prev_best[k-1][0] + current interval's weight

          new_end = current interval's end

          new_indices = prev_best[k-1][2] + [current interval's original index]

          Now, compare this new candidate with prev_best[k] (the current best for k before processing this interval):

          if new_weight > prev_best[k][0]:

              update best_candidate[k] to (new_weight, new_end, new_indices)

          elif new_weight == prev_best[k][0]:

              if new_end < prev_best[k][1]:

                  update

              elif new_end == prev_best[k][1]:

                  if new_indices is lex smaller than prev_best[k][2]:

                      update

   c. After processing all k for this interval, update the best_candidate array.

5. After processing all intervals, find the maximum weight among all k (0-4). Then, collect all candidates for those k that have this maximum weight. Among them, select the one with the lex smallest indices.

6. Return the indices array of the best candidate, sorted in lex order (but since the intervals are processed in order, the indices are added in the order of their selection, which is sorted by end time. Wait, but the lex order of the indices array is determined by the order of the indices. For example, [2,3] is lex smaller than [3,2]. So, when building the indices, the order in which the intervals are added matters. But since the intervals are processed in order of increasing end time, the indices are added in the order of their end times. So, the indices in the array are in the order of selection, which is sorted by end time. But the lex order of the array is determined by the order of the indices. For example, if two selections have the same weight and end time, but one has indices [1,3] and another [2,4], then [1,3] is lex smaller.

But when building the indices, the order is the order in which the intervals were added. So, the first interval added is the earliest in the sorted list, then the next, etc. So the indices in the array are in the order of the sorted intervals. So, the lex order of the array is determined by the order of the indices in the array. For example, if two arrays are [2,3] and [3,2], the first is lex smaller because 2 < 3 in the first position.

But when building the indices, the order is the order in which the intervals are added. So, the first interval in the selection is the earliest in the sorted list, then the next, etc. So the indices in the array are in the order of the sorted intervals. So, the lex order of the array is determined by the order of the indices in the array. For example, if two arrays are [2,3] and [3,2], the first is lex smaller because 2 < 3 in the first position.

But when comparing two candidates with the same weight and end time, the lex order of the indices array is determined by the order of the indices. So, the code needs to compare the indices arrays lexicographically.

So, in the code, when two candidates have the same weight and end time, we need to compare their indices arrays lex order.

Now, the problem is how to efficiently compute the latest j for each interval. Let's think about that.

The sorted_intervals list is sorted by end time. For each interval i in sorted_intervals, the start is l_i. We need to find the largest j < i where sorted_intervals[j].end <= l_i.

This can be done with binary search. The end times are in non-decreasing order. So, the list of end times is a sorted list. For each l_i, we can binary search the end times up to i-1 to find the largest end <= l_i.

So, for each interval in sorted_intervals, we can precompute the end times and perform a binary search.

In Python, the bisect module can be used. For example, the end times can be stored in a list, and for each l_i, we can use bisect_right to find the insertion point of l_i, then subtract 1 to get the largest j where end <= l_i.

But since the intervals are processed in order, and the end times are sorted, we can build a list of end times as we process the intervals. Wait, but the sorted_intervals are already sorted by end time, so the end times are in a list that is non-decreasing.

So, for each interval in sorted_intervals, the end times up to the current index are in non-decreasing order.

So, for the i-th interval in sorted_intervals (0-based), the end times of the first i intervals (0 to i-1) are in a list. So, for the current interval's start l_i, we can perform a binary search on the end times of the first i intervals (0 to i-1) to find the largest j where end <= l_i.

But how to get the end times up to i-1. Since the sorted_intervals are sorted by end time, the end times are in a list that is non-decreasing. So, for the i-th interval, the end times of the first i intervals (0 to i-1) are in a list. So, for example, for i=0, there are no intervals before, so j is -1. For i=1, the end times are [sorted_intervals[0].end], etc.

So, during processing, for each interval in sorted_intervals, we can keep a list of end times. For example, as we process each interval, we can append its end time to a list. Then, for the current interval, the end times of the previous intervals are the list up to the current index.

Wait, but the sorted_intervals are processed in order. So, for the i-th interval in sorted_intervals (0-based), the end times of the first i intervals (0 to i-1) are in a list. So, for example, when processing the 5th interval (i=4, 0-based), the end times of the first 4 intervals (0-3) are in the list.

So, during processing, we can maintain a list called end_times, which is built as we process each interval. For each interval in sorted_intervals, before processing it, we can append its end time to end_times. Then, when processing the next interval, the end_times list contains all end times up to the previous interval.

Wait, no. Because the intervals are processed in order. So, for the first interval (i=0), after processing, end_times will have its end. Then, for the next interval (i=1), when processing it, the end_times list contains the end of the first interval. So, when processing interval i, the end_times list contains the end times of the first i intervals (0 to i-1). So, for interval i, the end_times list is [e_0, e_1, ..., e_{i-1}].

So, during processing interval i, to find j, we can perform a binary search on end_times (up to i elements) to find the largest j where e_j <= l_i.

But how to handle this in code.

In Python, the bisect module's bisect_right function can be used. For example:

import bisect

end_times = []
for i in range(len(sorted_intervals)):
    l, r, w, idx = sorted_intervals[i]
    # find j where end_times[j] <= l
    j = bisect.bisect_right(end_times, l) - 1
    # process j
    end_times.append(r)

So, for each interval, we first compute j based on the end_times list (which contains the end times of the previous intervals), then append the current r to end_times.

This way, for each interval, the end_times list is built incrementally, and for the i-th interval, the end_times list contains the end times of the first i intervals (0 to i-1).

So, this approach works.

Now, putting it all together.

The code steps are:

1. Sort the intervals by end time, keeping track of their original indices.

2. Precompute the end_times list as we process each interval.

3. For each interval, compute j using bisect.

4. For each k from 4 down to 1, check if adding the current interval to the best_candidate_prev[k-1] (from the previous state) gives a better candidate for k.

5. Update the best_candidate array accordingly.

6. After processing all intervals, find the maximum weight and the best candidate.

Now, the code structure.

We need to represent the best_candidate for each k. Each best_candidate is a tuple of (max_weight, end_time, indices). We can represent this as a list of dictionaries or a list of objects, but in Python, tuples are easier.

So, best_candidate is a list of 5 elements (k=0 to 4). Each element is a tuple (max_weight, end_time, indices).

Initialize best_candidate as:

best_candidate = [
    (0, -infinity, []),
    (-infinity, infinity, []),
    (-infinity, infinity, []),
    (-infinity, infinity, []),
    (-infinity, infinity, []),
]

But in Python, we can represent -infinity as a very small number (like float('-inf')) and infinity as float('inf').

So, for k=0, the initial state is (0, -inf, []). For other k, ( -inf, inf, [] )

Then, for each interval in sorted_intervals:

   make a copy of best_candidate (prev_best)

   process the interval:

      l, r, w, idx = interval

      j = bisect.bisect_right(end_times, l) - 1

      for k in 4 down to 1:

          if j >=0 and prev_best[k-1][0] != float('-inf'):

              new_weight = prev_best[k-1][0] + w

              new_end = r

              new_indices = prev_best[k-1][2] + [idx]

              current_best = prev_best[k]

              if new_weight > current_best[0]:

                  best_candidate[k] = (new_weight, new_end, new_indices)

              elif new_weight == current_best[0]:

                  if new_end < current_best[1]:

                      best_candidate[k] = (new_weight, new_end, new_indices)

                  elif new_end == current_best[1]:

                      # compare lex order of new_indices and current_best[2]

                      if new_indices < current_best[2]:

                          best_candidate[k] = (new_weight, new_end, new_indices)

   append r to end_times.

Wait, but the end_times are appended after processing the interval. Because for the next interval, the end_times list should include the current interval's end. But during processing the current interval, the end_times list is the list of end times of the previous intervals. So, the code is correct.

But in the code above, after processing the interval, we append r to end_times. So, for the next interval, the end_times list includes the current interval's end.

Yes.

Now, after processing all intervals, we need to find the maximum weight among all k (0-4). Then, collect all candidates for those k that have this maximum weight, and select the one with the lex smallest indices.

So, after processing all intervals, iterate through k=0 to 4, find the maximum weight. Then, collect all candidates where the weight is equal to the maximum. Among these, select the one with the lex smallest indices.

If there are multiple candidates with the same weight and indices, the one with the earliest end time is better, but since the code already ensures that when weights and end times are equal, the lex smallest indices are chosen, the code should have the best candidate for each k.

So, the code steps after processing all intervals:

max_weight = max( best_candidate[k][0] for k in range(5) )

candidates = [ (best_candidate[k][2], best_candidate[k][0], best_candidate[k][1]) for k in range(5) if best_candidate[k][0] == max_weight ]

Then, among the candidates, select the one with the lex smallest indices.

But the candidates are tuples of (indices, weight, end_time). So, we can sort the candidates by the indices array lex order and pick the first one.

But in Python, comparing lists lexicographically is done automatically. So, for example, [2,3] < [3,2] is True.

So, the code can collect all the indices arrays of the candidates with max_weight, then select the smallest one.

But the candidates may have different k values. For example, the maximum weight could be achieved with k=2 and k=3. So, we need to collect all candidates (for all k) that have the max_weight, then select the one with the lex smallest indices.

So, the code would be:

candidates = []
for k in range(5):
    if best_candidate[k][0] == max_weight:
        candidates.append( best_candidate[k][2] )

# Now, find the lex smallest candidate
if not candidates:
    return []
# The candidates are lists of indices. Find the lex smallest.
candidates.sort()
result = candidates[0]

But wait, the candidates may have different lengths. For example, the maximum weight could be achieved with k=2 (two intervals) and k=3 (three intervals). But the problem says "at most 4". So, the code should return the lex smallest array among all possible candidates with the maximum weight, regardless of their length.

But according to the problem statement, the output is the lex smallest array of at most 4 indices. So, even if a candidate with k=2 has the same weight as a candidate with k=3, the lex smallest array (regardless of length) is chosen.

For example, if the candidates are [2,3] (weight 10) and [1,4,5] (weight 10), then [1,4,5] is longer but may be lex smaller. Wait, no. The lex order is determined by the first differing element. So, [1,4,5] is compared to [2,3] as follows: the first element 1 < 2, so [1,4,5] is lex smaller. So, even though it's longer, it's considered lex smaller.

But according to the problem statement, the array can be of any length up to 4. So, the code must return the lex smallest array among all possible candidates with the maximum weight, regardless of their length.

So, the code should collect all candidates (for all k) with the maximum weight, then select the lex smallest one.

But how to compare candidates of different lengths. For example, [1,2] and [1,2,3]. The first is shorter. According to the problem statement, the shorter array is lex smaller if the first min(len(a), len(b)) elements are the same.

So, for example, [1,2] is lex smaller than [1,2,3], because the first two elements are the same, and the first array is shorter.

So, when comparing two candidates, the code must first compare their elements up to the minimum length. If all elements are the same up to that point, the shorter array is considered smaller.

So, in Python, when comparing lists of different lengths, the shorter list is considered smaller if all elements up to the length of the shorter are equal.

For example:

[1,2] < [1,2,3] → True

[2,3] < [1,4,5] → True (since 2 > 1, so the second is smaller)

So, the code can collect all the candidates' indices arrays, then sort them lexicographically, and pick the first one.

So, the code would be:

candidates = [ best_candidate[k][2] for k in range(5) if best_candidate[k][0] == max_weight ]

candidates.sort()

return candidates[0] if candidates else []

But in the problem statement, the output is at most 4 indices. So, the code must return the lex smallest array among all possible candidates with maximum weight, which may have different lengths.

So, the code is correct.

Now, let's test this approach against the examples.

Example 1:

Input: intervals = [[1,3,2],[4,5,2],[1,5,5],[6,9,3],[6,7,1],[8,9,1]]

Sorted by end time:

Let's list the intervals with their end times:

Interval 0: [1,3,2] → end 3

Interval 1: [4,5,2] → end 5

Interval 2: [1,5,5] → end 5

Interval 3: [6,9,3] → end 9

Interval 4: [6,7,1] → end 7

Interval 5: [8,9,1] → end 9

Wait, no. The original input is:

intervals = [
 [1,3,2], → end 3
 [4,5,2], → end 5
 [1,5,5], → end 5
 [6,9,3], → end 9
 [6,7,1], → end 7
 [8,9,1], → end 9
]

When sorted by end time, the order is:

end 3 → interval 0

end 5 → interval 1 and 2 (since their end is 5). But how are they ordered? The start times are 4 and 1. So, when sorted by end, the order between intervals with the same end is determined by their start times. But the problem says to sort by end, but for intervals with the same end, the order may not matter. However, the code will process them in the order they appear in the sorted list. So, in this case, interval 1 (end 5) comes before interval 2 (end 5) because their end is the same, but their start is 4 and 1. Wait, no. The code sorts the intervals by end, but when ends are equal, the order is determined by the original order. Or perhaps the code uses the start time as a secondary key. But the problem statement doesn't specify. However, the code's sorted function will sort the intervals by end, and for those with the same end, their order is determined by their original positions. But in Python, the sorted function is stable if the key is the same. Wait, no. The sorted function in Python is not stable unless the key is the same. So, if two intervals have the same end, their order in the sorted list is determined by their other elements. For example, the tuple (l, r, w, idx) is sorted by r. So, for two intervals with the same r, the next key is l, then w, then idx. So, in the example, interval 1 has l=4, r=5, w=2, idx=1. Interval 2 has l=1, r=5, w=5, idx=2. So, when sorted by r, interval 1 comes before interval 2 because their r is the same, but l is 4 vs 1. So, interval 1 has l=4, which is larger than interval 2's l=1. So, when sorted by r, the order is interval 0 (r=3), then interval 1 (r=5, l=4), then interval 2 (r=5, l=1), then interval 4 (r=7), then interval 3 (r=9), then interval 5 (r=9). Wait, no. Because the code sorts the intervals by their end (r) in ascending order. So, the sorted list would be:

sorted_intervals = [
 (1,3,2,0),
 (4,5,2,1),
 (1,5,5,2),
 (6,7,1,4),
 (6,9,3,3),
 (8,9,1,5),
]

Wait, no. The end of interval 4 is 7, which is less than 9. So, the correct order after sorting by end is:

end 3 → interval 0

end 5 → interval 1 (end 5, l=4)

end 5 → interval 2 (end 5, l=1)

end 7 → interval 4 (end 7)

end 9 → interval 3 (end 9, l=6)

end 9 → interval 5 (end 9, l=8)

So, the sorted_intervals list is:

0: (1,3,2,0)

1: (4,5,2,1)

2: (1,5,5,2)

3: (6,7,1,4)

4: (6,9,3,3)

5: (8,9,1,5)

Now, let's process each interval.

Initialize best_candidate as:

k=0: (0, -inf, [])

k=1-4: (-inf, inf, [])

Processing interval 0 (l=1, r=3, w=2, idx=0):

end_times is empty. So, j = bisect_right([], 1) -1 → 0-1 = -1.

For k from 4 down to 1:

k=4: prev_best[3] is (-inf, inf, []). So, no change.

k=3: same.

k=2: same.

k=1: prev_best[0] is (0, -inf, []). new_weight = 0 + 2 = 2. new_end=3. new_indices = [0].

Compare with current best_candidate[1] (prev_best[1] is (-inf, inf, []). So, new_weight is better. So, best_candidate[1] becomes (2, 3, [0]).

So, after processing interval 0, best_candidate[1] is (2, 3, [0]).

end_times is now [3].

Processing interval 1 (l=4, r=5, w=2, idx=1):

end_times is [3]. bisect_right([3],4) → returns 1. j=0.

prev_best is the current best_candidate before processing this interval.

For k=4 down to 1:

k=4: prev_best[3] is (-inf, inf, []). No change.

k=3: prev_best[2] is (-inf, inf, []). No change.

k=2: prev_best[1] is (2,3, [0]). new_weight=2+2=4. new_end=5. new_indices = [0,1].

Compare with current best_candidate[2] (prev_best[2] is (-inf, inf, []). So, new_weight is better. So, best_candidate[2] becomes (4,5, [0,1]).

k=1: prev_best[0] is (0, -inf, []). new_weight=0+2=2. new_end=5. new_indices = [1]. Compare with current best_candidate[1] (2,3, [0]). new_weight is same as current. new_end is 5 > 3. So, current is better. No change.

So, after processing interval 1, best_candidate[2] is (4,5, [0,1]).

end_times is now [3,5].

Processing interval 2 (l=1, r=5, w=5, idx=2):

bisect_right([3,5], 1) → returns 0. j=-1.

For k=4 down to 1:

k=4: prev_best[3] is (-inf, inf, []). No change.

k=3: prev_best[2] is (4,5, [0,1]). new_weight=4+5=9. new_end=5. new_indices = [0,1,2].

Compare with current best_candidate[3] (prev_best[3] is (-inf, inf, []). So, new_weight is better. So, best_candidate[3] becomes (9,5, [0,1,2]).

k=2: prev_best[1] is (2,3, [0]). new_weight=2+5=7. new_end=5. new_indices = [0,2]. Compare with current best_candidate[2] (4,5, [0,1]). new_weight 7>4. So, update. best_candidate[2] becomes (7,5, [0,2]).

k=1: prev_best[0] is (0, -inf, []). new_weight=0+5=5. new_end=5. new_indices = [2]. Compare with current best_candidate[1] (2,3, [0]). new_weight 5>2. So, update. best_candidate[1] becomes (5,5, [2]).

So, after processing interval 2, best_candidate[1] is (5,5, [2]), best_candidate[2] is (7,5, [0,2]), best_candidate[3] is (9,5, [0,1,2]).

end_times is now [3,5,5].

Processing interval 3 (l=6, r=9, w=3, idx=3):

bisect_right([3,5,5],6) → returns 3. j=2.

prev_best is the current best_candidate before processing this interval.

For k=4 down to 1:

k=4: prev_best[3] is (9,5, [0,1,2]). new_weight=9+3=12. new_end=9. new_indices = [0,1,2,3]. Compare with current best_candidate[4] (prev_best[4] is (-inf, inf, []). So, new_weight is better. So, best_candidate[4] becomes (12,9, [0,1,2,3]).

k=3: prev_best[2] is (7,5, [0,2]). new_weight=7+3=10. new_end=9. new_indices = [0,2,3]. Compare with current best_candidate[3] (9,5, [0,1,2]). new_weight 10>9. So, update. best_candidate[3] becomes (10,9, [0,2,3]).

k=2: prev_best[1] is (5,5, [2]). new_weight=5+3=8. new_end=9. new_indices = [2,3]. Compare with current best_candidate[2] (7,5, [0,2]). new_weight 8>7. So, update. best_candidate[2] becomes (8,9, [2,3]).

k=1: prev_best[0] is (0, -inf, []). new_weight=0+3=3. new_end=9. new_indices = [3]. Compare with current best_candidate[1] (5,5, [2]). new_weight 3 <5. No change.

So, after processing interval 3, best_candidate[4] is (12,9, [0,1,2,3]), best_candidate[3] is (10,9, [0,2,3]), best_candidate[2] is (8,9, [2,3]).

end_times is now [3,5,5,9].

Processing interval 4 (l=6, r=7, w=1, idx=4):

bisect_right([3,5,5,9],6) → returns 3. j=2.

prev_best is the current best_candidate before processing this interval.

For k=4 down to 1:

k=4: prev_best[3] is (10,9, [0,2,3]). new_weight=10+1=11. new_end=7. new_indices = [0,2,3,4]. Compare with current best_candidate[4] (12,9, [0,1,2,3]). new_weight 11 <12. No change.

k=3: prev_best[2] is (8,9, [2,3]). new_weight=8+1=9. new_end=7. new_indices = [2,3,4]. Compare with current best_candidate[3] (10,9, [0,2,3]). new_weight 9 <10. No change.

k=2: prev_best[1] is (5,5, [2]). new_weight=5+1=6. new_end=7. new_indices = [2,4]. Compare with current best_candidate[2] (8,9, [2,3]). new_weight 6 <8. No change.

k=1: prev_best[0] is (0, -inf, []). new_weight=0+1=1. new_end=7. new_indices = [4]. Compare with current best_candidate[1] (5,5, [2]). new_weight 1 <5. No change.

So, no changes.

end_times is now [3,5,5,9,7].

Processing interval 5 (l=8, r=9, w=1, idx=5):

bisect_right([3,5,5,9,7],8) → the end_times list is [3,5,5,9,7]. Wait, no. After processing interval 4, the end_times list is [3,5,5,9,7]. So, when processing interval 5, the end_times list is [3,5,5,9,7]. But the end_times list is built by appending the current interval's end after processing it. So, during processing interval 5, the end_times list is [3,5,5,9,7], and the current interval's end is 9.

So, bisect_right([3,5,5,9,7], 8) → the list is [3,5,5,9,7]. Wait, but the list is not sorted. Because after appending 7 (from interval 4), the list is [3,5,5,9,7], which is not sorted. But this is a problem.

Wait, no. Because the intervals are processed in order of their end times. So, the end_times list should be non-decreasing. But in this case, interval 4 has end 7, which is added after interval 3's end 9. So, the end_times list becomes [3,5,5,9,7], which is not sorted. This is a problem.

Ah, this is a mistake in the code. Because the code appends the current interval's end to end_times after processing it, but the current interval's end may be smaller than previous ends. But the code is supposed to process intervals in sorted order by end time. So, the end_times list should be non-decreasing. But in this example, interval 4 has end 7, which is added after interval 3's end 9. So, the end_times list is not sorted. This is a problem.

Wait, no. The code sorts the intervals by end time. So, the sorted_intervals list is ordered by end time. So, the end_times list is built by appending the end times of the intervals in the order of sorted_intervals. So, the end_times list should be non-decreasing. Because the intervals are sorted by end time, each new end is >= the previous ones.

But in the example, interval 3 has end 9, and interval 4 has end 7. But this is impossible because the intervals are sorted by end time. So, interval 4's end must be >= interval 3's end. But in the example, interval 3 is (6,9,3,3) → end 9. Interval 4 is (6,7,1,4) → end 7. So, this interval's end is 7, which is less than 9. But this would mean that the sorted_intervals list is not sorted by end time. So, this is a mistake in the example.

Wait, no. The original input for example 1 is:

intervals = [[1,3,2],[4,5,2],[1,5,5],[6,9,3],[6,7,1],[8,9,1]]

So, the intervals are:

0: [1,3,2] → end 3

1: [4,5,2] → end 5

2: [1,5,5] → end 5

3: [6,9,3] → end 9

4: [6,7,1] → end 7

5: [8,9,1] → end 9

When sorted by end time, the order is:

end 3 → interval 0

end 5 → interval 1 and 2 (end 5)

end 5 → interval 2 (end 5)

end 7 → interval 4 (end 7)

end 9 → interval 3 (end 9)

end 9 → interval 5 (end 9)

So, the sorted_intervals list is:

0: (1,3,2,0)

1: (4,5,2,1)

2: (1,5,5,2)

3: (6,7,1,4)

4: (6,9,3,3)

5: (8,9,1,5)

So, the end_times list after processing each interval is:

After interval 0: [3]

After interval 1: [3,5]

After interval 2: [3,5,5]

After interval 3: [3,5,5,7]

After interval 4: [3,5,5,7,9]

After interval 5: [3,5,5,7,9,9]

So, the end_times list is non-decreasing. So, the code is correct.

So, during processing interval 3 (end 7), the end_times list is [3,5,5,7]. Then, interval 4 is processed (end 9), so end_times becomes [3,5,5,7,9]. Then, interval 5 is processed (end 9), end_times becomes [3,5,5,7,9,9].

So, the code is correct.

Now, processing interval 5 (l=8, r=9, w=1, idx=5):

bisect_right(end_times, 8) → end_times is [3,5,5,7,9,9]. The value 8 is between 7 and 9. So, bisect_right returns 4. j=4-1=3.

prev_best is the current best_candidate before processing this interval.

For k=4 down to 1:

k=4: prev_best[3] is (10,9, [0,2,3]). new_weight=10+1=11. new_end=9. new_indices = [0,2,3,5]. Compare with current best_candidate[4] (12,9, [0,1,2,3]). new_weight 11 <12. No change.

k=3: prev_best[2] is (8,9, [2,3]). new_weight=8+1=9. new_end=9. new_indices = [2,3,5]. Compare with current best_candidate[3] (10,9, [0,2,3]). new_weight 9 <10. No change.

k=2: prev_best[1] is (5,5, [2]). new_weight=5+1=6. new_end=9. new_indices = [2,5]. Compare with current best_candidate[2] (8,9, [2,3]). new_weight 6 <8. No change.

k=1: prev_best[0] is (0, -inf, []). new_weight=0+1=1. new_end=9. new_indices = [5]. Compare with current best_candidate[1] (5,5, [2]). new_weight 1 <5. No change.

So, no changes.

After processing all intervals, the best_candidate for each k is:

k=0: (0, -inf, [])

k=1: (5,5, [2])

k=2: (8,9, [2,3])

k=3: (10,9, [0,2,3])

k=4: (12,9, [0,1,2,3])

The maximum weight is 12. So, the candidate is [0,1,2,3]. But according to the example 1, the correct output is [2,3], which has a weight of 5+3=8. But this contradicts. So, what's wrong here?

Wait, example 1's output is [2,3], which are the intervals with indices 2 and 3. The weights are 5 and 3, sum 8. But according to the code's calculation, the maximum weight is 12, which is achieved by selecting intervals 0,1,2,3 (indices 0,1,2,3). But their sum is 2+2+5+3=12. But in the example, the maximum is 5+3=8. So, this suggests a mistake in the code.

But this is impossible. So, there must be a mistake in the code's logic.

Ah, I see the problem. The code is considering the sum of the weights, but in the example, the correct answer is to select intervals 2 and 3 (indices 2 and 3), which have weights 5 and 3, sum 8. But according to the code's calculation, the maximum sum is 12. So, this suggests that the code is incorrect.

But why?

Looking back at the example 1:

The input intervals are:

0: [1,3,2]

1: [4,5,2]

2: [1,5,5]

3: [6,9,3]

4: [6,7,1]

5: [8,9,1]

The code's sorted_intervals are sorted by end time. So, the order is:

0: end 3

1: end 5

2: end 5

3: end 7

4: end 9

5: end 9

Wait, no. The code's sorted_intervals are sorted by end time. So, the end times are:

interval 0: 3

interval 1:5

interval 2:5

interval 3:7 (from [6,7,1])

interval 4:9 (from [6,9,3])

interval 5:9 (from [8,9,1])

So, the sorted_intervals list is:

0: (1,3,2,0)

1: (4,5,2,1)

2: (1,5,5,2)

3: (6,7,1,4)

4: (6,9,3,3)

5: (8,9,1,5)

So, during processing interval 3 (index 3 in sorted_intervals, which is [6,7,1,4]), the code's best_candidate[4] is updated to (12,9, [0,1,2,3]). But the sum of their weights is 2+2+5+1=10? Wait, no. The code's new_weight is prev_best[k-1].weight + current interval's weight.

Wait, for interval 3 (sorted_intervals[3] is [6,7,1,4], which has weight 1. So, when processing this interval, for k=4:

prev_best[3] is the best_candidate[3] before processing this interval. Let's see:

Before processing interval 3 (sorted_intervals[3]), the best_candidate[3] is (7,5, [0,2]). So, when processing interval 3 (k=4):

new_weight = 7 + 1 = 8. new_end=7. new_indices = [0,2,4]. But this is for k=4. So, the code's best_candidate[4] is updated to (8,7, [0,2,4]).

Wait, no. Let's re-examine the processing of interval 3.

Interval 3 is sorted_intervals[3], which is (6,7,1,4). So, l=6, r=7, w=1, idx=4.

When processing this interval:

prev_best is the best_candidate before processing this interval.

For k=4 down to 1:

k=4: prev_best[3] is (7,5, [0,2]). new_weight=7+1=8. new_end=7. new_indices = [0,2,4]. Compare with current best_candidate[4] (prev_best[4] is (-inf, inf, []). So, new_weight is better. So, best_candidate[4] becomes (8,7, [0,2,4]).

k=3: prev_best[2] is (8,9, [2,3]). new_weight=8+1=9. new_end=7. new_indices = [2,3,4]. Compare with current best_candidate[3] (prev_best[3] is (7,5, [0,2]). new_weight 9>7. So, update. best_candidate[3] becomes (9,7, [2,3,4]).

k=2: prev_best[1] is (5,5, [2]). new_weight=5+1=6. new_end=7. new_indices = [2,4]. Compare with current best_candidate[2] (8,9, [2,3]). new_weight 6 <8. No change.

k=1: prev_best[0] is (0, -inf, []). new_weight=0+1=1. new_end=7. new_indices = [4]. Compare with current best_candidate[1] (5,5, [2]). new_weight 1 <5. No change.

So, after processing interval 3, best_candidate[4] is (8,7, [0,2,4]), best_candidate[3] is (9,7, [2,3,4]).

Then, processing interval 4 (sorted_intervals[4] is (6,9,3,3), l=6, r=9, w=3, idx=3).

bisect_right(end_times,6) → end_times up to this point are [3,5,5,7,9]. So, bisect_right([3,5,5,7,9],6) → returns 3. j=2.

prev_best is the current best_candidate before processing this interval.

For k=4 down to 1:

k=4: prev_best[3] is (9,7, [2,3,4]). new_weight=9+3=12. new_end=9. new_indices = [2,3,4,3]. But the indices are [2,3,4,3], which is invalid because the same index is used twice. Wait, no. The current interval's index is 3. So, new_indices is [2,3,4,3]. But this is not possible because the same index is used twice. So, this is a mistake.

Ah, here's the problem. The code is using the original indices of the intervals, but when processing the intervals in sorted order, the same original index can be used multiple times. But the problem requires that the selected indices are distinct. So, the code is incorrect because it allows the same index to be selected multiple times.

Oh no! This is a critical error. The code is not checking whether the current interval's index is already in the indices array. So, the code is allowing duplicate indices, which is invalid.

So, the code is incorrect. The problem requires that the selected indices are distinct. So, the code must ensure that each interval is selected at most once.

This is a major flaw in the approach. The code currently allows the same interval to be selected multiple times, which is invalid.

So, the code needs to be modified to track which intervals are selected and ensure that each is used at most once.

But how to do this efficiently.

This is a major problem. The previous approach is incorrect because it allows the same interval to be selected multiple times. So, the code needs to be adjusted to prevent this.

So, the code must track the selected indices and ensure that each is unique.

But how to model this in the DP state.

The previous approach tracks the indices in the selection, but it doesn't check for duplicates. So, the code is incorrect.

So, the code needs to be modified to ensure that each interval is selected at most once.

But how to do this.

The problem is that the code's current approach allows the same interval to be added multiple times, which is invalid.

So, the code must track the selected indices and ensure that they are unique.

But with the current approach, the code is not doing that.

So, the code is incorrect.

This is a major issue. So, the previous approach is invalid.

So, what's the solution?

The solution is to track the selected indices and ensure that they are unique. But how to do this efficiently.

One way is to track the selected indices in the DP state. But with 5e4 intervals, this is impossible.

Alternative approach: since the intervals are processed in order, and each interval is considered once, the code can track the selected indices as a set. But this is not feasible for large N.

Alternative idea: since the intervals are processed in order, and the code is building the selection in the order of the sorted intervals, the code can track the selected indices as a list, and when adding a new interval, check if its index is already in the list.

But this would require O(k) time for each addition, which is acceptable since k is up to 4.

So, in the code, when creating new_indices, we need to check if the current interval's index is already present in the previous indices. If it is, we skip adding it.

But how to do this.

In the code, when creating new_indices, we can check if the current index (idx) is in prev_best[k-1][2]. If it is, then we cannot add it again. So, the new_indices would be invalid.

But this would require checking for the presence of the current index in the previous indices. For example, when processing interval i, which has index idx, and the previous indices are [0,1,2], if idx is 2, then adding it would create a duplicate.

So, the code must ensure that the current interval's index is not already in the previous indices.

So, the code needs to be modified to check for this.

So, in the code, when creating new_indices, we need to check if the current idx is already in prev_best[k-1][2]. If it is, then we cannot add this interval, and the new candidate is invalid.

So, the code would have:

if idx in prev_best[k-1][2]:
    continue
else:
    new_indices = prev_best[k-1][2] + [idx]

But this would require O(k) time for each check, which is acceptable since k is up to 4.

So, this is a necessary modification to the code.

But this changes the approach significantly. So, the code must be modified to include this check.

So, the code would look like:

for k in 4 down to 1:

    j = bisect.bisect_right(end_times, l) - 1

    if j >=0 and prev_best[k-1][0] != float('-inf'):

        if idx in prev_best[k-1][2]:

            continue

        new_weight = prev_best[k-1][0] + w

        new_end = r

        new_indices = prev_best[k-1][2] + [idx]

        # compare and update best_candidate[k]

So, this would prevent adding the same interval multiple times.

But this introduces a problem: the code may miss some valid combinations where the same interval is not added multiple times.

But in the example 1, the code's previous approach allowed adding interval 3 (index 3) multiple times, leading to an incorrect sum.

So, with this modification, the code would correctly track the selected indices.

But this modification requires checking for the presence of the current index in the previous indices.

So, let's re-examine example 1 with this modification.

Example 1:

After processing interval 3 (sorted_intervals[3], which is [6,7,1,4], idx=4):

prev_best[3] is (7,5, [0,2]).

When processing k=4:

prev_best[3] is (7,5, [0,2]). new_weight =7+1=8. new_end=7. new_indices = [0,2,4]. Since 4 is not in [0,2], it's added. So, best_candidate[4] becomes (8,7, [0,2,4]).

Then, processing interval 4 (sorted_intervals[4], which is [6,9,3,3], idx=3):

bisect_right(end_times,6) → j=2.

prev_best[3] is (9,7, [2,3,4]).

For k=4:

prev_best[3] is (9,7, [2,3,4]). new_weight=9+3=12. new_end=9. new_indices = [2,3,4,3]. But idx=3 is already in [2,3,4], so this is skipped.

So, no update for k=4.

For k=3:

prev_best[2] is (8,9, [2,3]). new_weight=8+3=11. new_end=9. new_indices = [2,3,3]. But idx=3 is already in [2,3], so this is skipped.

For k=2:

prev_best[1] is (5,5, [2]). new_weight=5+3=8. new_end=9. new_indices = [2,3]. Since 3 is not in [2], it's added. So, new_indices is [2,3]. Compare with current best_candidate[2] (8,9, [2,3]). new_weight is same. new_end is same. compare indices. new_indices is [2,3], which is same as current. So, no change.

So, after processing interval 4, best_candidate[2] remains (8,9, [2,3]).

Processing interval 5 (sorted_intervals[5], which is [8,9,1,5], idx=5):

bisect_right(end_times,8) → j=3 (end_times up to this point are [3,5,5,7,9,9]. So, bisect_right([3,5,5,7,9,9],8) → returns 4. j=3.

prev_best[3] is (9,7, [2,3,4]).

For k=4:

prev_best[3] is (9,7, [2,3,4]). new_weight=9+1=10. new_end=9. new_indices = [2,3,4,5]. Since 5 is not in [2,3,4], it's added. Compare with current best_candidate[4] (8,7, [0,2,4]). new_weight 10>8. So, best_candidate[4] becomes (10,9, [2,3,4,5]).

For k=3:

prev_best[2] is (8,9, [2,3]). new_weight=8+1=9. new_end=9. new_indices = [2,3,5]. Compare with current best_candidate[3] (9,7, [2,3,4]). new_weight 9 ==9. new_end 9>7. So, current best_candidate[3] is better. No change.

For k=2:

prev_best[1] is (5,5, [2]). new_weight=5+1=6. new_end=9. new_indices = [2,5]. Compare with current best_candidate[2] (8,9, [2,3]). new_weight 6 <8. No change.

For k=1:

prev_best[0] is (0, -inf, []). new_weight=0+1=1. new_end=9. new_indices = [5]. Compare with current best_candidate[1] (5,5, [2]). new_weight 1 <5. No change.

So, after processing all intervals, the best_candidate for each k is:

k=0: (0, -inf, [])

k=1: (5,5, [2])

k=2: (8,9, [2,3])

k=3: (9,7, [2,3,4])

k=4: (10,9, [2,3,4,5])

The maximum weight is 10. But according to example 1, the correct maximum weight is 8 (5+3=8). So, this suggests that the code is still incorrect.

Wait, but in example 1, the correct answer is to select intervals 2 and 3 (indices 2 and 3), which have weights 5 and 3, sum 8. But according to the code's current calculation, the maximum weight is 10, achieved by selecting intervals 2,3,4,5 (indices 2,3,4,5). But their sum is 5+3+1+1=10. But this is possible only if the intervals are non-overlapping.

Let's check the intervals:

Interval 2: [1,5,5] → index 2.

Interval 3: [6,7,1] → index 4.

Interval 4: [6,9,3] → index 3.

Interval 5: [8,9,1] → index 5.

Wait, no. The sorted_intervals list is:

0: (1,3,2,0)

1: (4,5,2,1)

2: (1,5,5,2)

3: (6,7,1,4)

4: (6,9,3,3)

5: (8,9,1,5)

So, the indices in the sorted_intervals are 0,1,2,4,3,5.

So, when the code selects indices 2,3,4,5, these are the original indices of the intervals in the sorted_intervals list. So, the original indices are:

sorted_intervals[2].idx → 2

sorted_intervals[3].idx →4

sorted_intervals[4].idx →3

sorted_intervals[5].idx →5

So, the selected intervals are:

2: [1,5,5] (original index 2)

4: [6,7,1] (original index 4)

3: [6,9,3] (original index 3)

5: [8,9,1] (original index 5)

These intervals are:

[1,5], [6,7], [6,9], [8,9]

But [6,7] and [6,9] overlap. So, they are overlapping. So, the code's selection is invalid.

Ah, this is the problem. The code is allowing overlapping intervals to be selected.

But according to the problem statement, the selected intervals must be non-overlapping.

So, the code's current approach is incorrect because it allows overlapping intervals to be selected.

So, the code must ensure that the intervals in the selection are non-overlapping.

But how to ensure this.

The code's current approach uses the end time of the last interval in the selection to ensure that the next interval's start is after the end of the last. But this is only valid if the intervals are processed in order of their start times. But the code processes intervals in order of their end times.

So, the code's current approach is incorrect.

The problem is that the code's current approach allows adding intervals that overlap with previous intervals in the selection, as long as their start is after the end of the last interval in the selection.

But this is not correct. Because the selection is built by adding intervals in the order of their end times, but the intervals may have overlapping with earlier intervals in the selection.

For example, in the current selection [2,3,4,5], the intervals are:

2: [1,5]

3: [6,9]

4: [6,7]

5: [8,9]

But interval 3 starts at 6, which is after interval 2's end at 5. So, they are non-overlapping. Interval 4 starts at 6, which is before interval 3's end at 9. So, interval 4 and 3 overlap. So, the code's selection is invalid.

So, the code's current approach is incorrect because it allows adding intervals that overlap with previous intervals in the selection.

The code's current approach only checks that the current interval's start is after the previous interval's end. But this is not sufficient to ensure that all intervals in the selection are non-overlapping.

For example, if the selection has intervals A, B, C, processed in order of their end times, and B's start is after A's end, but C's start is before B's end, then A and C may overlap.

So, the code's current approach is incorrect.

The correct approach must ensure that all intervals in the selection are non-overlapping, not just consecutive ones.

So, the code's current approach is incorrect.

This is a major problem. The code's current approach only ensures that each new interval does not overlap with the previous one, but allows earlier intervals to overlap with later ones.

So, the code's approach is incorrect.

The correct approach must ensure that all intervals in the selection are non-overlapping.

So, the code's current approach is incorrect.

This requires a different approach.

The correct approach must track the entire set of selected intervals and ensure that they are all non-overlapping. But with 5e4 intervals and up to 4 selections, this is challenging.

Alternative idea: the code's current approach is correct if the intervals are processed in order of their start times. Because then, when adding a new interval, it's guaranteed that it doesn't overlap with any previous intervals in the selection.

But how?

If the intervals are processed in order of their start times, then when adding a new interval, its start is >= the start of all previous intervals. But this doesn't prevent overlapping.

Alternatively, if the intervals are processed in order of their end times, and when adding a new interval, it's guaranteed that it doesn't overlap with any previous intervals in the selection.

But this is not the case. For example, if the selection contains an interval that ends at 5, and a new interval starts at 4 and ends at 6, then they overlap.

So, the code's current approach is incorrect.

The correct approach must ensure that all intervals in the selection are non-overlapping. So, the code's current approach is incorrect.

This is a major problem. So, the code's current approach is incorrect.

So, what's the correct approach?

The correct approach is to select a subset of intervals that are non-overlapping, and the sum of their weights is maximized. This is similar to the classic weighted interval scheduling problem, but with a maximum of 4 intervals.

The classic approach for the weighted interval scheduling problem with k intervals is to use dynamic programming where the state is the number of intervals selected and the last end time. For each interval, you can choose to include it (if it doesn't overlap with the previous intervals) or not.

But the code's current approach only checks the last interval's end time, which is insufficient.

So, the correct approach is to track the last end time and ensure that each new interval's start is after the last end time. This way, all intervals in the selection are non-overlapping.

But this is exactly what the code's current approach is doing. So, why is the code's selection invalid?

Because the code's current approach allows adding intervals in any order, but the selection is built in the order of the sorted intervals (sorted by end time). So, when adding a new interval, it's checked against the last interval in the selection. But this ensures that all intervals in the selection are non-overlapping.

Wait, no. Because the selection is built in the order of the sorted intervals (sorted by end time). So, when adding a new interval, it's added after the previous intervals in the selection. But the new interval's start is after the previous interval's end. So, all intervals in the selection are non-overlapping.

For example, in the selection [A, B, C], where A is processed first (earlier end), then B is added (start after A's end), then C is added (start after B's end). So, all are non-overlapping.

But in the example where the code's selection is [2,3,4,5], the intervals are:

2: [1,5] (end 5)

3: [6,9] (end 9)

4: [6,7] (end 7)

5: [8,9] (end 9)

But interval 3's start is 6, which is after interval 2's end (5). So, they are non-overlapping. Interval 4's start is 6, which is before interval 3's end (9). So, interval 4 and 3 overlap. So, the code's selection is invalid.

But according to the code's approach, when adding interval 4 (sorted_intervals[3], which is [6,7,1,4], idx=4), the code checks that its start (6) is after the previous interval's end (5). So, it's allowed. Then, when adding interval 3 (sorted_intervals[4], which is [6,9,3,3], idx=3), the code checks that its start (6) is after the previous interval's end (7). But 6 <7, so it's not allowed. So, why is this allowed?

Ah, here's the mistake. The code's sorted_intervals are sorted by end time. So, interval 3 (sorted_intervals[3]) has end 7, and interval 4 (sorted_intervals[4]) has end 9. So, when processing interval 4 (sorted_intervals[4]), the code's end_times list is [3,5,5,7,9]. So, when processing interval 4 (sorted_intervals[4]), the code's bisect_right
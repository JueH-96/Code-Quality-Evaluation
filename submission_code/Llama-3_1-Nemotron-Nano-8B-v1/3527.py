<think>
Okay, I need to solve this problem where there's a circular arrangement of red and blue tiles, and I have to process queries that either update a tile's color or ask for the number of alternating groups of a certain size. Hmm, let's think about how to approach this.

First, let's understand what an alternating group is. An alternating group is a contiguous subset (in the circular arrangement) where each tile (except the first and last) alternates in color. So, for example, if the group is of size 3, the first and third can be the same, but the middle must be different. But since the group is contiguous and part of a circle, the first and last elements of the group are adjacent in the group, so they must alternate as well. Wait, no. Wait, the definition says each tile in the group except the first and last has different colors from adjacent tiles in the group. Wait, the group is a contiguous subset. So for a group of size k, the first and last are adjacent in the group. So for the group to be alternating, the entire sequence must alternate. So for example, a group of size 4 must be like 0,1,0,1 or 1,0,1,0. Because each consecutive pair alternates, and the first and last (since it's a group) must also alternate. Wait, no. Wait, the group is contiguous, but the group's first and last elements are adjacent in the group. So for the group to be alternating, the entire sequence must alternate. So for any group of size k, the colors must alternate between each consecutive element, including the first and last. Because the group is a contiguous subset, but in the circular arrangement, the first and last of the group are adjacent. So for example, a group of size 3 must have colors like 0,1,0 or 1,0,1. Because each consecutive pair alternates, and the first and third (since they are adjacent in the group) must also alternate. Wait, but in a group of size 3, the first and third are adjacent in the group. So the group's first and third must be different. So the group must be a sequence where each consecutive element is different, and the first and last (since they are adjacent) are also different. So for a group of size k, the entire sequence must be a proper alternating sequence of length k. So for example, a group of size 2 can be any two consecutive tiles that are different. Because the first and second are adjacent, and the group is of size 2, so the two must alternate. So for a group of size 2, the two tiles must be different. For a group of size 3, the three tiles must alternate, like 0,1,0. For size 4, 0,1,0,1, etc.

So the problem is, for each query of type 1 (ask for count of groups of size s), we need to find how many such groups exist in the current color arrangement.

But how do we compute this efficiently, especially given that the colors can be updated, and the array is circular?

The naive approach would be, for each query of type 1, to check all possible starting positions and see if the next s-1 elements form an alternating group. But with colors.length up to 5e4 and queries up to 5e4, this would be O(Q*N) time, which would be 2.5e9 operations, which is way too slow.

So we need a smarter way to track the alternating groups and update the counts efficiently when the colors change.

Hmm. Let's think about the structure of the alternating groups. Each group is determined by its starting position and its length. But since the array is circular, the starting positions wrap around.

An alternative approach is to model the array as a sequence of runs. For example, a run is a maximal sequence of consecutive same colors. For example, in the example 1, after the first update, colors become [0,0,1,0,1]. The runs are 0 (length 2), 1 (1), 0 (1), 1 (1). Then, the alternating groups are those that start and end at the boundaries of these runs. Because between runs, the color changes. So, for example, a group of size 2 can be formed between two adjacent runs of different colors. But how does this help?

Wait, perhaps the key is that an alternating group can only start at the boundaries of runs. Because if there's a run longer than 1, then any group that starts inside the run (except the first element) would have consecutive same colors, which would break the alternating condition.

Wait, let's think. Suppose we have a run of red tiles of length 3: R R R. Then, any group that starts at the first R and includes the next R would have two consecutive Rs, which is invalid. So the only possible groups that can start in a run are those that start at the first element of the run and have a length that allows them to end at the next run. So, perhaps the valid groups are those that start at the boundaries between runs and have a length that allows them to alternate.

So, for example, if the runs are R R B B R, then the boundaries are between R and B, B and R, etc. So groups can start at the R-B boundary, and then alternate. So the group would be R B R, which is valid. But if the group is longer than the available runs, then it's invalid.

So, the idea is that the valid groups are those that start at a run boundary and have a length that allows them to alternate through the runs.

But how to model this? Let's think of the runs as a list. For example, the runs can be represented as a list of tuples (color, length). Then, the number of possible groups of size s is determined by the transitions between runs.

But how?

Alternatively, for each position in the array, we can precompute the maximum possible group that starts at that position and alternates. Then, for a given s, we can check how many positions have a maximum group length >= s, and the group of size s is possible.

But this approach would require, for each position, knowing the maximum possible group length starting there. However, when a color is updated, this information can change for nearby positions, which complicates things.

Alternatively, perhaps we can model the array as a sequence of transitions between colors. For each position i, we can track the next position where the color changes. Then, the maximum possible group starting at i is the distance to the next color change. But again, this might not capture all possibilities.

Wait, let's think of the array as a circular array. For each position i, the maximum possible group that starts at i and alternates is determined by the next positions. For example, if the color at i is 0, then the next color must be 1, then 0, etc. So the group can be as long as possible until the color doesn't alternate. But since the array is circular, the group can wrap around.

But this seems complicated. Let's think of the array as a linear array, but with the first and last elements connected. So, for a group of size s, the starting position can be any index, and the group wraps around if necessary.

But how to model this efficiently.

Another approach: For each possible group size s, the number of valid groups is equal to the number of positions where a group of size s starting at that position is valid.

But how to compute this quickly.

Let's think of the array as a circular array. For each position i, the group starting at i with size s is valid if, for all 0 <= k < s-1, colors[i +k] != colors[i +k+1 mod n], and also colors[i + s-1 mod n] != colors[i mod n]. Wait, no. Because the group is contiguous and of size s. So the group is i, i+1, ..., i+s-1 (mod n). For the group to be alternating, each consecutive pair in the group must alternate. So for each j from 0 to s-2, colors[i +j] != colors[i +j+1 mod n]. Also, the first and last elements of the group (i and i+s-1 mod n) must be different. Because the group is contiguous, the first and last elements are adjacent in the group. So the entire group must form a proper alternating sequence.

Wait, no. Wait, the group is a contiguous subset. So the group is a sequence of s elements in a row. For the group to be alternating, each consecutive pair in the group must alternate. So for the group to be valid, all consecutive pairs in the group must be different. So for a group of size s, there are s-1 consecutive pairs. So for example, a group of size 2 must have two elements of different colors. A group of size 3 must have three elements where each consecutive pair is different, which implies that the first and third are the same. Wait, no. For a group of size 3, the first and second are different, the second and third are different. So the first and third can be the same. For example, 0,1,0. Then, the consecutive pairs are 0-1 (different), 1-0 (different). So the group is valid. But the first and third are the same. But since the group is contiguous, the first and third are not adjacent in the group. Wait, no. The group is of size 3, so the elements are positions 0,1,2. The consecutive pairs are (0,1), (1,2). The first and third (0 and 2) are not consecutive in the group, so their colors can be the same. So the group is valid. So the group is valid as long as all consecutive pairs in the group are different. The first and last elements of the group (positions 0 and 2) are adjacent in the group only if the group is of size 2. Wait, no. For a group of size 3, the group is positions 0,1,2. The consecutive pairs are (0,1), (1,2). The first and last elements (0 and 2) are not consecutive in the group. So their colors can be the same. So the group is valid as long as all consecutive pairs in the group are different.

Wait, but the problem statement says that the group is a contiguous subset of tiles in the circle with alternating colors (each tile in the group except the first and last one has a different color from its adjacent tiles in the group). Wait, that's a different condition. So the first and last tiles in the group can be the same, but all the tiles in between must alternate. So for example, a group of size 3 can be 0,1,0. The first and last are same, but the middle is different. So this is allowed. But a group of size 4 must have alternating colors, so 0,1,0,1. Because each consecutive pair is different, and the first and last (0 and 1) are different. So for a group of size s, the entire sequence must alternate between colors, which implies that the first and last elements are different if s is even, and same if s is odd.

Wait, no. Let's see: For a group of size s, the consecutive pairs must alternate. So for s even, the first and last elements are different. For s odd, they are the same. Because each step alternates. For example:

s=2: 0,1 → different.

s=3: 0,1,0 → same.

s=4: 0,1,0,1 → different.

So the group is valid if all consecutive pairs in the group are different, regardless of the first and last. Because the problem statement says that each tile in the group except the first and last must have different colors from adjacent tiles. Wait, the problem statement says: "each tile in the group except the first and last one has a different color from its adjacent tiles in the group." So the first and last can have the same color as their adjacent tiles in the group. Wait, no. Let me re-read the problem statement.

The problem statement says: "An alternating group is a contiguous subset of tiles in the circle with alternating colors (each tile in the group except the first and last one has a different color from its adjacent tiles in the group)."

So, the first and last tiles in the group can have the same color as their adjacent tiles in the group. Wait, no. Because the group is contiguous. For example, in a group of size 3, the first tile's adjacent in the group is the second. The last tile's adjacent in the group is the second to last. So the first and last tiles are not adjacent in the group. So the condition is that all tiles except the first and last must have different colors from their adjacent tiles in the group. So the first tile's adjacent is the second, which must be different. The last tile's adjacent is the second to last, which must be different. But the first and last can be same or different. So for a group of size 3, the first and last can be same. For example, 0,1,0 is allowed. But 0,0,1 is not allowed because the first two are same. So the group is valid if all consecutive pairs in the group are different, except possibly the first and last. Wait, no. Because the group is contiguous, the first and last are adjacent in the group only if the group is of size 2. For example, group of size 3: the first and last are not adjacent. So the condition is that each tile except the first and last must have different colors from their adjacent tiles. So the first tile's next is the second, which must be different. The second's next is the third, which must be different. The third's previous is the second, which must be different. So for a group of size 3, the first and third can be same. So the group is valid as long as consecutive pairs are different.

So the group is valid if all consecutive pairs in the group are different. Because the first and last are not adjacent in the group (unless the group is size 2). So for a group of size s, the condition is that for all 0 <= i < s-1, colors[i] != colors[i+1 mod s]. Wait, no. Because the group is contiguous. For example, group of size 3: positions 0,1,2. The consecutive pairs are (0,1), (1,2). The group is valid if 0 !=1 and 1 !=2. The first and last (0 and 2) can be same. So the group is valid if all consecutive pairs in the group are different.

So the problem reduces to finding all contiguous sequences of length s where all consecutive elements are different. And since the array is circular, the group can wrap around.

So for a query of type 1, given s, we need to count the number of starting positions i (0 <= i < n) such that the group starting at i and of length s has all consecutive elements different.

But how to compute this efficiently, especially with updates.

Hmm. Let's think of the array as a circular array. For each position i, we can precompute the maximum possible group length starting at i. For example, the maximum group length starting at i is the maximum k such that for all 0 <= j < k-1, colors[i+j] != colors[i+j+1 mod n]. Then, for a query of size s, the count is the number of positions i where the maximum group length starting at i is >= s.

But this approach would require, for each position, knowing the maximum possible group length starting there. However, when a color is updated, this information can change for positions near the updated index, which complicates things.

Alternatively, perhaps we can model the array as a collection of runs. A run is a maximal sequence of consecutive same colors. For example, in the array [0,0,1,0,1], the runs are 0 (length 2), 1 (1), 0 (1), 1 (1). Then, the maximum group length starting at a run boundary is determined by the next runs. For example, starting at the first run (0), the next run is 1. So the maximum group length here is 2 (0,1). But if the next run is longer, like 1,1, then the group can be longer. Wait, no. Because the group must have all consecutive elements different. So the group can only be as long as the sum of the lengths of the runs until a run of the same color is encountered. Wait, perhaps not. Let's think.

If the runs are 0 (length 2), 1 (length 3), 0 (length 1), then the maximum group starting at the first run (0) would be 2 (0,1). Because after the first run, the next run is 1, which is different. So the group can be 0,1,0 (if the next run is 0). Wait, but the next run after 1 is 0. So the group would be 0 (from first run), 1 (from second run), 0 (from third run). So the group length is 2 (from first run) + 3 (from second run) + 1 (from third run) = 6? No, that's not right. Because the group is contiguous. So the group starts at the first run's start (position 0) and includes the next runs until the color changes again. Wait, perhaps the maximum group length starting at a run boundary is the sum of the lengths of the runs until a run of the same color is found. For example, if the runs are 0 (a), 1 (b), 0 (c), then the maximum group starting at a is a + b + c, because the colors alternate. But if the runs are 0 (a), 1 (b), 1 (c), then the group can only be a + b, because after the 1 run of length b, the next run is also 1, which is same as the previous. So the group can't include that.

So the idea is that the maximum group length starting at a run boundary is the sum of the lengths of consecutive runs that alternate in color. So, for example, if the runs are 0,1,0,1, then the maximum group length starting at the first 0 is the sum of all runs (since they alternate). But if there's a run of the same color in between, the group can't extend beyond that.

So, the maximum group length starting at a run boundary is the sum of the lengths of the runs in the longest possible alternating sequence starting from that run.

But how to model this. Let's think of the runs as a list. For example, runs = [ (color, length), ... ].

Then, for each run i, the maximum group length starting at run i is the sum of the lengths of runs i, i+1, i+2, ... until a run with the same color as run i is encountered. Because once you hit a run of the same color, the group can't include it, as the colors would not alternate.

Wait, but the group is a contiguous sequence of tiles. So, for example, if run i is 0 (length a), run i+1 is 1 (length b), run i+2 is 0 (length c), then the group starting at run i can include a + b + c tiles. Because the colors alternate 0,1,0. So the group is valid as long as all consecutive elements are different. So the group can be of length a + b + c. But if run i+2 is 1 (length c), then the group can only be a + b, because after that, the next run is 1, which is same as run i+1. So the group can't include that.

So the maximum group length starting at run i is the sum of the lengths of runs i, i+1, i+2, ... until a run with the same color as run i is found. Because after that, the color would repeat, breaking the alternating condition.

But how to compute this for all runs. Let's model the runs as a list, and for each run, precompute the next run that has the same color. Then, the maximum group length for run i is the sum of the lengths from run i to the run before the next same color.

But this approach would require O(n) space for the runs and their next same color pointers. However, when a color is updated, the runs list can change, which complicates things.

But given that the array is circular, the runs list is a circular list as well. So, for example, if the array is [0,0,1,0,1], the runs are 0 (2), 1 (1), 0 (1), 1 (1), and then it wraps around to the first run again.

But maintaining the runs list dynamically with updates is challenging. Each update can split a run into up to three runs (if the color changes in the middle of a run). For example, if a run is 0,0,0 and we change the middle element to 1, the run is split into 0,1,0. So the runs list would need to be updated accordingly.

This seems possible, but the complexity is high. However, given the constraints (colors.length up to 5e4 and queries up to 5e4), we need an O(1) or O(log n) per query approach.

Alternatively, perhaps we can model the array as a binary array (0s and 1s) and track for each position the length of the longest possible alternating group starting at that position. Then, for a query of size s, the answer is the number of positions where the maximum group length is >= s.

But how to compute and update these maximum group lengths efficiently.

Let's think of the array as a circular array. For each position i, the maximum group length starting at i is the maximum k such that for all 0 <= j < k-1, colors[i+j] != colors[i+j+1 mod n]. So, for example, if the array is 0,1,0,1, then the maximum group length for each position is 4. If the array is 0,0,1,0,1, then the maximum group lengths are:

i=0: 2 (0,0 is invalid, but wait, no. Wait, the group starts at i=0. The first two elements are 0 and 0, which are same. So the maximum group length is 1. Because the group can't have any consecutive same colors. Wait, no. The group must have all consecutive elements different. So for a group starting at i=0, the first element is 0. The next element is 0, which is same. So the group can't be longer than 1. So the maximum group length for i=0 is 1.

But this seems to contradict the earlier example. So perhaps the maximum group length for a position i is the maximum k such that the group starting at i and of length k has all consecutive elements different.

So, for each position i, the maximum group length is the maximum possible k where for all 0 <= j < k-1, colors[i+j] != colors[i+j+1 mod n]. So, for the array [0,0,1,0,1], the maximum group lengths would be:

i=0: 1 (since next element is same)
i=1: 1 (next is 1, same)
i=2: 2 (1,0)
i=3: 2 (0,1)
i=4: 1 (1,0)

But this approach would require, for each position, to compute the maximum possible group length. However, when a color is updated, this can affect the maximum group lengths of nearby positions. For example, changing the color at position 1 from 0 to 0 (no change) does nothing. But changing it to 1 would split the run and affect the maximum group lengths of positions 0,1,2.

So, how can we model this efficiently?

An alternative approach is to precompute for each position i the next position where the color changes. For example, for each i, next_diff[i] is the smallest j > i such that colors[j] != colors[i]. Then, the maximum group length starting at i is next_diff[i] - i. But this is only valid if the next_diff[i] is the next position where the color changes. However, this approach would not account for the circular nature of the array. Also, when the array is updated, the next_diff array would need to be updated for multiple positions, which could be expensive.

Alternatively, perhaps we can model the array as a linked list of runs, where each run is a sequence of same colors. Then, for each run, the maximum group length starting at the first element of the run is the sum of the lengths of the runs that follow until a run of the same color is encountered. For example, if the runs are 0 (a), 1 (b), 0 (c), then the maximum group length starting at the first run is a + b + c. But if the next run after 0 (c) is 1 (d), then the group can continue. So the maximum group length is the sum of all runs until a run of the same color as the starting run is found.

But how to manage this with updates. Each update can split a run into up to three parts. For example, if a run is 0 (length 5) and we change the middle element to 1, the run is split into 0 (2), 1 (1), 0 (2). So the runs list would need to be updated accordingly.

This seems manageable. Let's think about how to represent the runs. We can have a list of runs, each represented as (color, start, end), where start and end are the indices in the original array. The runs are stored in a circular list, so the last run's end is followed by the first run's start.

When a color is updated at index i, we need to find which run contains i. Then, split that run into up to three runs (before the change, the run is split into left, changed, right). Then, check the neighboring runs to see if they can be merged.

For example, suppose the original run is from a to b, color c. We change index i to d. Then, the run is split into a to i-1 (c), i (d), i+1 to b (c). Then, check if the previous run (before a) has color d, and the next run (after i+1 to b) has color d. If so, merge them.

But this requires maintaining a linked list of runs, which can be done with a dictionary or a list. However, for large n (5e4), this could be manageable if the number of runs is not too high. But in the worst case, each element is a separate run, leading to O(n) runs, which would make operations O(n) time, which is not acceptable.

So this approach may not be efficient enough.

Alternative idea: For each possible group size s, the number of valid groups is equal to the number of positions where the group of size s starting at that position has all consecutive elements different. So, for each s, we can precompute the count and update it when the colors change.

But how to track this for all s. Since s can be up to n, this is not feasible.

Another approach: For each possible s, the valid groups are those where the group is a run of s elements with alternating colors. So, for example, a group of size 2 is valid if the two elements are different. A group of size 3 is valid if the three elements are 0,1,0 or 1,0,1. So, the group must alternate between colors.

But how to model this. Let's think of the array as a sequence of transitions between colors. For each position i, we can track the length of the longest possible alternating group starting at i. Then, for a query of size s, the answer is the number of positions i where the maximum group length starting at i is >= s.

But how to compute and update this maximum group length.

Let's model the array as a circular array. For each position i, the maximum group length starting at i is the maximum k such that for all 0 <= j < k-1, colors[i+j] != colors[i+j+1 mod n]. So, the maximum group length is the maximum possible k where the next k-1 elements alternate in color.

This can be computed using a sliding window approach, but for each query, this would take O(n) time, which is not feasible.

Alternatively, we can precompute for each position i the maximum possible group length starting at i. Let's call this array max_len. Then, for a query of size s, the answer is the number of i where max_len[i] >= s.

The problem is how to compute max_len efficiently and update it when a color changes.

Let's think about how to compute max_len initially. For each position i, we can compute the maximum possible group length starting at i by checking consecutive elements until a same color is found. For example:

max_len[i] = 1
for j in i+1 to i + max_possible:
    if colors[j] != colors[j-1]:
        max_len[i] +=1
    else:
        break

But this is O(n^2) time, which is not feasible for n=5e4.

So we need a way to compute max_len in O(n) time.

Wait, but the array is circular. So perhaps we can model the max_len array using a dynamic programming approach.

For example, in a linear array, the max_len[i] can be computed as follows:

if colors[i] == colors[i+1], then max_len[i] = 1
else, max_len[i] = 1 + max_len[i+1]

But this is for a linear array. However, for a circular array, this approach would not work directly. Also, the max_len[i] depends on the next elements, which may wrap around.

Alternatively, for a linear array, the max_len can be computed in O(n) time by processing from the end. For example:

n = len(colors)
max_len = [1] * n
for i in range(n-2, -1, -1):
    if colors[i] != colors[i+1]:
        max_len[i] = max_len[i+1] + 1
    else:
        max_len[i] = 1

But this is for a linear array. For a circular array, this approach would not account for the wrap-around. For example, if the array is [0,1,0,1], the max_len for each position would be 4. But the above code would compute max_len[0] as 1 + max_len[1], which is 1 + (1 + max_len[2]) = 1 + (1 + (1 + max_len[3])) = 4. But for a circular array, the code would not work because the last element's next is the first element.

So, for a circular array, the code would need to handle the wrap-around. But how?

Alternatively, we can double the array to handle the circularity. For example, create a new array that is colors + colors, and then compute the max_len for the first n elements, considering the wrap-around. But this would take O(n) time and O(n) space.

But even this approach would not work for the dynamic case where colors are updated. Because each update would require recomputing the max_len array, which is O(n) time per update, which is not feasible for 5e4 updates.

So this approach is not suitable for the problem.

Hmm. So we need a data structure that can handle dynamic updates and allow us to query the number of positions where the maximum group length is >= s.

Another idea: For each position i, the maximum group length starting at i is the length of the longest possible run of alternating colors starting at i. So, for example, if the colors are 0,1,0,1, then each position's max_len is 4. If the colors are 0,0,1,0,1, then the max_len for each position is:

i=0: 1 (next is 0)
i=1: 1 (next is 1)
i=2: 2 (1,0)
i=3: 2 (0,1)
i=4: 1 (1,0)

So, the max_len array is [1,1,2,2,1].

Now, when a color is updated, say, position 1 is changed from 0 to 1, the array becomes [0,1,1,0,1]. The max_len array becomes:

i=0: 1 (next is 1)
i=1: 1 (next is 1)
i=2: 2 (1,0)
i=3: 2 (0,1)
i=4: 1 (1,0)

Wait, no. After changing position 1 to 1, the array is [0,1,1,0,1]. So:

i=0: next is 1 (different), so max_len[0] = 1 + max_len[1]. But max_len[1] is 1 (since next is 1). So max_len[0] = 2? But the group starting at 0 is 0,1, which is valid. So the max_len[0] should be 2.

But according to the previous approach, the code would compute max_len[0] as 1 + max_len[1]. But max_len[1] is 1, so max_len[0] is 2. But in reality, the group starting at 0 is 0,1, which is valid. So the max_len[0] is 2.

But in the example given in the problem statement, after the first query (changing index 1 to 0), the array becomes [0,0,1,0,1]. The max_len array would be:

i=0: next is 0 (same) → max_len[0] =1
i=1: next is 0 (same) → max_len[1] =1
i=2: next is 1 (different) → max_len[2] =1 + max_len[3]
i=3: next is 0 (different) → max_len[3] =1 + max_len[4]
i=4: next is 1 (different) → max_len[4] =1 + max_len[0] (which is 1) → 2.

But this would create a loop. So the dynamic programming approach for a linear array doesn't work for a circular array.

So this suggests that the dynamic programming approach is not suitable for the circular case.

Alternative idea: For each position i, the maximum group length starting at i is the maximum k such that the next k-1 elements are all different from their previous. So, for each i, we can compute this by checking the next elements until a same color is found. But this is O(n) per query, which is not feasible.

But perhaps we can precompute for each position i the next position where the color changes. For example, next_diff[i] is the smallest j > i such that colors[j] != colors[i]. Then, the maximum group length starting at i is next_diff[i] - i. But this is only valid if the next_diff[i] is the next position where the color changes. However, this approach would not account for the circular nature of the array. Also, when the array is updated, the next_diff array would need to be updated for multiple positions, which could be expensive.

But how to compute next_diff[i]. For example, in the array [0,0,1,0,1], next_diff for each position is:

i=0: next_diff is 2 (color 1)
i=1: next_diff is 2 (color 1)
i=2: next_diff is 3 (color 0)
i=3: next_diff is 4 (color 1)
i=4: next_diff is 0 (color 0, since it's circular)

So the maximum group lengths would be:

i=0: 2-0=2 → but the group starting at 0 is 0,0, which is invalid. So this approach is not correct.

Ah, right. Because next_diff[i] is the first position where the color changes. So the group from i to next_diff[i]-1 is all the same color. So the maximum group length starting at i is next_diff[i] - i. But this is the length of the run starting at i. But the group must have all consecutive elements different. So this approach is not suitable.

So this suggests that the next_diff approach is not helpful.

Alternative idea: For each position i, the maximum group length starting at i is the length of the longest possible sequence of alternating colors starting at i. So, for example, if the colors are 0,1,0,1, then each position's max_len is 4. If the colors are 0,0,1,0,1, then the max_len for each position is 1,1,2,2,1.

But how to compute this efficiently.

Let's think of the array as a circular array. For each position i, the maximum group length is the maximum k such that for all 0 <= j < k-1, colors[i+j] != colors[i+j+1 mod n]. So, the group is valid if all consecutive pairs in the group are different.

This is equivalent to finding the longest possible run of alternating colors starting at i. So, for each i, the maximum group length is the maximum k where the sequence i, i+1, ..., i+k-1 (mod n) has all consecutive elements different.

To compute this, we can model the array as a graph where each node is a position, and an edge from i to i+1 exists if colors[i] != colors[i+1]. Then, the maximum group length starting at i is the length of the longest path starting at i in this graph. But this is a directed acyclic graph (DAG) since each step moves forward. However, the circular nature complicates things.

But even if we model it as a graph, finding the longest path for each node is O(n) time, which is not feasible for dynamic updates.

Another approach: For each position i, the maximum group length is the minimum between the length of the run starting at i and the maximum possible group length that can be formed by alternating colors.

But I'm not sure.

Alternatively, perhaps we can precompute for each position i the maximum group length starting at i using a sliding window approach. For example, for each i, we can check the next elements until a same color is found. But this is O(n^2) time, which is not feasible.

Hmm. This is getting complicated. Let's think of the problem differently. For each query of type 1, we need to count the number of groups of size s. A group of size s is valid if all consecutive elements in the group are different. So, for a group of size s, the number of valid groups is equal to the number of positions i where the group starting at i and of length s has all consecutive elements different.

But how to compute this quickly.

Let's think of the array as a circular array. For each possible s, the valid groups are those where the group is a sequence of s elements with all consecutive elements different. So, for example, for s=2, the valid groups are pairs of consecutive elements that are different. For s=3, the valid groups are triplets where each consecutive pair is different.

But how to count this for all s.

An alternative idea is to precompute for each possible s the count of valid groups. But since s can be up to n, this is not feasible.

But perhaps we can precompute for each position i the maximum possible group length starting at i, and then for each query s, count the number of positions i where the maximum group length is >= s.

But how to compute the maximum group length for each i.

Let's think of the array as a circular array. For each position i, the maximum group length is the maximum k such that for all 0 <= j < k-1, colors[i+j] != colors[i+j+1 mod n]. So, the group is valid if all consecutive elements are different.

This is equivalent to finding the longest possible sequence of alternating colors starting at i. So, for example, if the array is 0,1,0,1, then each position's maximum group length is 4. If the array is 0,0,1,0,1, then the maximum group lengths are 1,1,2,2,1.

But how to compute this efficiently.

Let's think of the array as a circular array. For each position i, the maximum group length is the length of the longest possible sequence of alternating colors starting at i. This can be computed using a sliding window approach, but for each query, this would take O(n) time, which is not feasible.

But perhaps we can precompute for each position i the maximum group length and maintain this information dynamically.

But how to update the maximum group length when a color changes.

When a color at position i is updated, it can affect the maximum group lengths of positions i-1, i, and i+1 (mod n). Because changing the color at i can break or create alternating sequences that include i.

For example, if the array is [0,1,0,1], and we change position 2 to 0, the array becomes [0,1,0,1]. Wait, no. Wait, changing position 2 from 0 to 0 does nothing. But if we change it to 1, the array becomes [0,1,1,1]. Then, the maximum group lengths for positions 0,1,2,3 would be 1,1,1,1.

So, changing a color can have a ripple effect on the maximum group lengths of nearby positions.

This suggests that maintaining the maximum group lengths dynamically is challenging, as each update can affect O(1) positions, but each of those positions may require checking their neighbors, leading to O(1) or O(k) time per update, where k is the number of positions affected.

But given the constraints of 5e4 queries, we need an O(1) or O(log n) per update approach.

Another idea: For each position i, the maximum group length is determined by the next positions. So, if we can model the maximum group length for each position as the minimum between the current run and the next run's maximum group length plus one.

But I'm not sure.

Alternatively, perhaps we can model the maximum group length for each position i as follows:

max_len[i] = 1 if colors[i] == colors[i+1 mod n] else 1 + max_len[i+1 mod n]

But this is a recursive relation. However, this would create a cycle in the circular array, leading to infinite recursion. So this approach is not feasible.

Hmm. This is getting really stuck. Let's think of the problem from another angle.

The key observation is that a valid group of size s must have all consecutive elements different. So, for a group of size s, the entire sequence must alternate between colors. This means that the group's length s must be such that the colors alternate every step. So, for example, a group of size 3 must have colors like 0,1,0 or 1,0,1. A group of size 4 must have 0,1,0,1 or 1,0,1,0.

So, for a group of size s, the colors must alternate every step. This implies that the group's colors must form a sequence where each consecutive pair is different. So, the group is valid if and only if the entire sequence is a proper alternating sequence.

So, the problem reduces to counting the number of positions i where the group starting at i and of size s is a proper alternating sequence.

But how to compute this efficiently.

Let's think of the array as a circular array. For each position i, we can precompute the maximum possible group length starting at i. Then, for a query of size s, the answer is the number of positions i where the maximum group length is >= s.

But how to compute and update the maximum group length.

Another idea: The maximum group length for a position i is the length of the longest possible alternating sequence starting at i. This can be computed using a sliding window approach, but for dynamic updates, this is not feasible.

Alternatively, we can model the array as a collection of intervals where the colors alternate. For example, each interval is a maximal sequence of alternating colors. Then, for each interval, we can track its length. When a color is updated, the intervals around the updated position are split and merged as needed. Then, for a query of size s, the answer is the number of intervals of length >= s.

But this approach may not work because the intervals are not directly related to the group sizes. For example, a group of size 3 can be part of a longer interval.

Alternatively, perhaps the valid groups are exactly the intervals of alternating colors. So, the number of valid groups of size s is the number of intervals of length >= s.

But this is not correct. For example, consider an interval of length 4 (0,1,0,1). This interval contains groups of size 2, 3, and 4. So, the number of valid groups of size 2 in this interval is 4 (each possible starting position in the interval). But according to the interval approach, the interval has length 4, so it contributes 4-2+1 =3 groups of size 2. But this is not correct.

Wait, no. For an interval of length 4, the valid groups of size 2 are the first two elements, the next two, etc. But in the interval 0,1,0,1, the groups of size 2 are:

0,1 → valid

1,0 → valid

0,1 → valid

1,0 → valid (since it's circular)

So there are 4 groups of size 2. But the interval length is 4, and 4-2+1=3. So this approach is not correct.

So the interval approach is not directly applicable.

Hmm. This is really challenging. Let's think of the problem in terms of the example given.

Example 1:

After the first query, colors = [0,0,1,0,1]. The query is to count groups of size 4.

The valid groups are:

0,0,1,0 → invalid (first two are same)

0,1,0,1 → valid (starts at index 1)

1,0,1,0 → valid (starts at index 2)

0,1,0,0 → invalid (last two are same)

1,0,0,1 → invalid (middle two are same)

0,0,1,0 → invalid (first two are same)

Wait, but the array is circular. So the groups are:

Indices 0-3: 0,0,1,0 → invalid.

Indices 1-4: 0,1,0,1 → valid.

Indices 2-0: 1,0,1,0 → valid.

Indices 3-1: 0,1,0,0 → invalid.

Indices 4-2: 1,0,0,1 → invalid.

So there are 2 valid groups of size 4. Which matches the example's output.

So the valid groups are those where all consecutive elements are different. So for a group of size 4, the four elements must alternate.

So, the problem is to count the number of positions i where the group starting at i and of size s has all consecutive elements different.

But how to compute this efficiently.

Let's think of the array as a circular array. For each position i, we can precompute the maximum group length starting at i. Then, for a query of size s, the answer is the number of positions i where the maximum group length is >= s.

But how to compute the maximum group length.

Another idea: For each position i, the maximum group length is the maximum k such that for all 0 <= j < k-1, colors[i+j] != colors[i+j+1 mod n]. So, the group is valid if all consecutive elements are different.

This can be represented as a binary array where each position i has a value of 1 if colors[i] != colors[i+1 mod n], else 0. Then, the maximum group length starting at i is the maximum number of consecutive 1's starting at i, plus 1. Because each 1 represents a valid transition, and the group length is the number of transitions plus 1.

For example, if the transitions are 1,1,0, then the group length is 2 (transitions 1 and 1) → group length 3? No. Wait, each transition is between consecutive elements. So, for a group of length k, there are k-1 transitions. So, if there are m consecutive 1's in the transitions array starting at i, then the maximum group length is m+1.

So, the problem reduces to finding, for each position i, the maximum m such that the transitions array has m consecutive 1's starting at i. Then, the maximum group length is m+1.

The transitions array is of length n, where transitions[i] is 1 if colors[i] != colors[i+1 mod n], else 0.

So, for example, in the array [0,0,1,0,1], the transitions array is:

0 (0 and 0) → 0

0 and 1 → 1

1 and 0 → 1

0 and 1 → 1

1 and 0 → 1

So transitions = [0,1,1,1,1]

Then, the maximum group lengths are:

i=0: transitions[0] is 0 → group length 1.

i=1: transitions[1] is 1, transitions[2] is 1, transitions[3] is 1, transitions[4] is 1, transitions[0] is 0. So the maximum consecutive 1's starting at i=1 is 4. So group length is 4+1=5? But the array is of length 5. So the group starting at i=1 is 0,1,0,1,0. Which has transitions 1,1,1,1. So group length 5. But the array is circular, so the group wraps around. But the group of size 5 would have all consecutive elements different. So the maximum group length for i=1 is 5.

But according to the transitions array, the consecutive 1's starting at i=1 are 4 (transitions[1], 2, 3, 4), then transitions[0] is 0. So the maximum consecutive 1's is 4. So group length is 4+1=5.

Yes. So this approach seems correct.

So, the maximum group length starting at i is equal to the number of consecutive 1's in the transitions array starting at i, plus 1.

So, the problem now reduces to:

- Maintain a transitions array where transitions[i] is 1 if colors[i] != colors[i+1 mod n], else 0.

- For each query of type 1 (count groups of size s), the answer is the number of positions i where the maximum consecutive 1's in transitions starting at i is >= s-1.

- For each query of type 2 (update color at index i), we need to update the transitions array at positions i-1 and i (mod n), since changing colors[i] affects transitions[i-1] and transitions[i].

So, the key is to maintain for each position i the maximum number of consecutive 1's in the transitions array starting at i. Then, for a query of size s, the answer is the number of positions i where this maximum is >= s-1.

But how to maintain this information efficiently.

This seems more manageable. Because each update to the transitions array affects at most two positions (i-1 and i). So, when a color is updated, we need to recompute transitions[i-1] and transitions[i], and then update the maximum consecutive 1's information for the affected positions.

But how to compute the maximum consecutive 1's for each position.

The maximum consecutive 1's starting at i can be computed using a sliding window approach. For example, for each i, the maximum consecutive 1's is the maximum k such that transitions[i], transitions[i+1], ..., transitions[i+k-1] are all 1.

But maintaining this for dynamic updates is challenging.

An alternative approach is to precompute for each position i the maximum consecutive 1's to the right. This can be done using a dynamic programming approach. For example:

max_consec_ones[i] = transitions[i] ? max_consec_ones[i+1] + 1 : 0

But this is for a linear array. For a circular array, this approach would not work directly. However, we can model the transitions array as a circular array and compute max_consec_ones for each position.

But when the transitions array is updated, the max_consec_ones array needs to be recomputed, which is O(n) time per update. This is not feasible.

So, we need a data structure that can handle dynamic updates and allow us to query the maximum consecutive 1's starting at each position.

This sounds like a problem that can be solved with a segment tree. Each node in the segment tree can store information about the maximum consecutive 1's in the segment, as well as the prefix and suffix counts of consecutive 1's. This way, when a segment is updated, the segment tree can combine the information from the left and right children to compute the maximum consecutive 1's for the parent.

For example, each node in the segment tree can store:

- max_consec: the maximum number of consecutive 1's in the segment.

- prefix: the number of consecutive 1's starting at the left end of the segment.

- suffix: the number of consecutive 1's ending at the right end of the segment.

- total: the total number of 1's in the segment (not needed for this problem).

When merging two children, the parent's max_consec is the maximum of the left child's max_consec, the right child's max_consec, and the left child's suffix + right child's prefix.

The prefix of the parent is the left child's prefix if the left child's prefix covers the entire left segment and the left child's suffix is equal to the left segment's length. Otherwise, it's the left child's prefix.

Similarly for the suffix.

This way, the segment tree can efficiently compute the maximum consecutive 1's in any segment, and handle updates in O(log n) time.

Once we have this data structure, for each position i, the maximum consecutive 1's starting at i can be found by querying the segment tree for the maximum consecutive 1's in the range [i, i + k - 1], but this is not directly possible. Alternatively, the segment tree can be used to find the maximum consecutive 1's starting at i by querying the range [i, i + max_possible] and finding the maximum.

But this approach may not directly give the maximum consecutive 1's starting at i. However, the segment tree can be augmented to store for each node the maximum consecutive 1's starting at the left end of the segment. This way, when querying the segment tree for the range starting at i, we can get the maximum consecutive 1's starting at i.

Alternatively, the segment tree can be designed to answer the maximum consecutive 1's starting at a given position i. This can be done by storing for each node the maximum consecutive 1's starting at the left end of the segment, and the maximum consecutive 1's starting at any position within the segment.

But this is getting complicated.

Alternatively, for each position i, the maximum consecutive 1's starting at i is the maximum k such that transitions[i], transitions[i+1], ..., transitions[i+k-1] are all 1. This can be represented as the maximum length of a run of 1's starting at i.

To compute this, we can use a segment tree where each node stores the maximum run of 1's in the segment, the maximum run starting at the left, the maximum run ending at the right, and the total length of the segment. This allows us to merge segments efficiently.

For example, when querying the maximum run of 1's starting at position i, we can traverse the segment tree, starting from the root and checking if the left child contains position i. If the left child's left segment includes i, we check if the left child's prefix (run of 1's starting at the left) is contiguous and covers up to the end of the left child. If so, we can take the left child's prefix and continue into the right child.

This approach would allow us to compute the maximum consecutive 1's starting at i in O(log n) time.

But implementing this requires a custom segment tree that can handle these queries.

Once we have this data structure, for each query of type 1 (size s), we need to count the number of positions i where the maximum consecutive 1's starting at i is >= s-1.

But how to count this efficiently.

This is another problem. We need to count the number of positions i where the maximum consecutive 1's starting at i is >= s-1.

This can be done by maintaining a frequency array that tracks, for each possible value of consecutive 1's, the number of positions i with that value. But since the maximum possible consecutive 1's is up to n, this would require O(n) space, which is not feasible.

Alternatively, we can use a binary indexed tree (Fenwick tree) or a segment tree to track the count of positions i where the maximum consecutive 1's starting at i is >= s-1.

But this would require maintaining a data structure that can answer the count of elements >= x in a dynamic array. This can be done with a frequency array and a prefix sum array, but for dynamic updates, a binary indexed tree or a segment tree is needed.

But this would require O(n) space for the frequency array, which is feasible for n up to 5e4.

So, the plan is:

1. Maintain a transitions array where transitions[i] is 1 if colors[i] != colors[i+1 mod n], else 0.

2. Maintain a segment tree that can answer the maximum consecutive 1's starting at any position i.

3. Maintain a frequency array freq where freq[k] is the number of positions i where the maximum consecutive 1's starting at i is k.

4. For each query of type 1 (size s), the answer is the sum of freq[k] for k >= s-1.

5. For each query of type 2 (update color at i), we need to:

   a. Update the colors array.

   b. Update the transitions array at positions i-1 and i (mod n).

   c. For each of these positions (i-1 and i), update the transitions value.

   d. For each of these positions, recompute the maximum consecutive 1's starting at i-1 and i, and update the frequency array accordingly.

But how to recompute the maximum consecutive 1's starting at i-1 and i.

This is where the segment tree comes into play. When transitions[i-1] or transitions[i] is updated, we can query the segment tree for the maximum consecutive 1's starting at i-1 and i, and update the frequency array accordingly.

But this approach would require O(log n) time per update for each of the two positions, and O(1) time to update the frequency array.

But maintaining the frequency array is challenging because when a position's maximum consecutive 1's changes, we need to decrement the old count and increment the new count.

So, the steps for handling an update are:

When updating transitions[j], for j in {i-1, i}:

   old_max = previous maximum consecutive 1's starting at j.

   new_max = query the segment tree for the new maximum consecutive 1's starting at j.

   decrement freq[old_max] by 1.

   if new_max != old_max:

       increment freq[new_max] by 1.

But how to track the previous maximum consecutive 1's for each position j.

We need to maintain an array max_consec_ones that stores the current maximum consecutive 1's starting at each position j. This array is updated whenever transitions[j] or any of the transitions that affect j's consecutive 1's are updated.

But how to compute max_consec_ones[j] after an update.

This is where the segment tree comes into play. When transitions[j] is updated, we can query the segment tree for the maximum consecutive 1's starting at j to get the new max_consec_ones[j].

So, the steps for handling an update are:

For each query of type 2 (update index i to color c):

   old_color = colors[i]

   colors[i] = c

   # Update transitions[i-1] and transitions[i]

   j1 = (i-1) % n

   j2 = i % n

   for j in {j1, j2}:

       old_trans = transitions[j]

       new_trans = 0 if colors[j] == colors[(j+1) % n] else 1

       if new_trans != old_trans:

           # Update transitions[j]

           transitions[j] = new_trans

           # Query the new max_consec_ones for j and j+1, etc.

           # But how?

           # The maximum consecutive 1's starting at j may have changed.

           # So, we need to recompute max_consec_ones[j]

           # But how to do this efficiently.

           # We can query the segment tree for the maximum consecutive 1's starting at j.

           new_max = query_segment_tree(j, j)

           # Update max_consec_ones[j]

           if max_consec_ones[j] != new_max:

               # Decrement freq[max_consec_ones[j]]

               freq[max_consec_ones[j]] -= 1

               # Increment freq[new_max]

               freq[new_max] += 1

               max_consec_ones[j] = new_max

But this approach is not correct. Because changing transitions[j] can affect the maximum consecutive 1's not only at j but also at j-1, j+1, etc. For example, if transitions[j] is changed from 0 to 1, then the maximum consecutive 1's starting at j-1 may increase.

So, this approach would miss those changes.

This suggests that the frequency array approach is not sufficient, as a single update can affect multiple positions' max_consec_ones.

This makes the problem even more challenging.

Alternative idea: Instead of maintaining the frequency array, for each query of type 1, we can iterate through all positions and count how many have max_consec_ones[i] >= s-1. But this is O(n) time per query, which is not feasible for 5e4 queries.

So, this approach is not suitable.

Hmm. This is really challenging. Let's think of another way.

The key insight is that the maximum consecutive 1's starting at i is the maximum k such that transitions[i], transitions[i+1], ..., transitions[i+k-1] are all 1. So, the maximum consecutive 1's starting at i is the length of the longest run of 1's starting at i.

This can be represented as a linked list where each node points to the next position in the run of 1's. For example, if transitions[i] is 1 and transitions[i+1] is 1, then the run starting at i includes i+1, and so on.

But maintaining this linked list dynamically is possible. When a transition changes, we can update the linked list for the affected positions.

But this approach would require O(1) time per update for each affected position, but in the worst case, a single update can affect O(n) positions, which is not feasible.

So, this approach is not suitable.

Another idea: For each position i, the maximum consecutive 1's starting at i is the maximum k such that transitions[i], transitions[i+1], ..., transitions[i+k-1] are all 1. This can be represented as a run of 1's. So, the maximum consecutive 1's starting at i is the length of the run that includes i.

So, if we can track all runs of 1's in the transitions array, then for each run of length L, the number of positions i in the run where the maximum consecutive 1's starting at i is k is L - k + 1 for each k from 1 to L.

But this is not directly helpful for counting the number of positions i with max_consec_ones[i] >= s-1.

For example, a run of length L contributes (L - (s-1) + 1) = L - s + 2 positions i where max_consec_ones[i] >= s-1, but only if L >= s-1.

So, the total number of valid positions is the sum over all runs of max(0, L - s + 2) if L >= s-1.

But how to track the runs of 1's in the transitions array and their lengths.

This seems promising. Because when a transition is updated, we can split or merge runs, and track their lengths.

So, the plan is:

- Maintain a list of runs of 1's in the transitions array. Each run is represented as a start and end index, and a length.

- For each query of type 1 (size s), the answer is the sum over all runs of max(0, run.length - s + 2) if run.length >= s-1.

- For each query of type 2 (update index i), we need to update the transitions array and adjust the runs accordingly.

But how to manage the runs.

When a transition is updated, it can split a run into up to three parts (before the transition, the transition itself, and after). For example, if a run is from a to b, and the transition at position j (within a to b) is changed from 1 to 0, then the run is split into a to j-1 and j+1 to b. If the transition at j is changed from 0 to 1, then we need to check if the neighboring transitions are 1 and merge runs as needed.

This approach requires maintaining the runs in a data structure that allows efficient splitting and merging. A balanced binary search tree or a linked list with pointers can be used. However, for large n, this can be manageable if the number of runs is not too high.

But in the worst case, each transition is 0, leading to O(n) runs, which would make operations O(n) time, which is not feasible.

So, this approach may not be efficient enough.

But given the time constraints, perhaps this is the best approach to proceed with.

So, the steps are:

1. Initialize the transitions array and compute the initial runs of 1's.

2. For each query of type 1, compute the sum over all runs of max(0, run.length - s + 2) if run.length >= s-1.

3. For each query of type 2:

   a. Update the colors array.

   b. Update the transitions array at positions i-1 and i.

   c. For each of these positions, check if the transition has changed.

   d. For each changed transition, update the runs:

      i. If the transition was 0 and is now 1:

         - Check the previous and next transitions. If they are 1, merge the runs.

      ii. If the transition was 1 and is now 0:

         - Split the current run into up to three parts.

   e. Update the runs list accordingly.

But implementing this requires a lot of code and careful handling of the runs.

But given the time constraints, perhaps this is the way to go.

So, let's outline the code.

First, we need to represent the runs. We can use a list of tuples (start, end, length), where start and end are the indices in the transitions array. The length is end - start + 1.

But maintaining this list in a way that allows efficient splitting and merging is challenging.

Alternatively, we can use a linked list to represent the runs. Each node contains the start and end indices of the run, and pointers to the previous and next nodes.

When a transition is updated, we can find the run that contains the transition index and split or merge as needed.

But implementing a linked list in Python is possible, but may be time-consuming.

Alternatively, we can use a list to represent the runs and manage it with binary search.

But this may not be efficient enough.

Another idea: For each transition index j, track which run it belongs to. Then, when j is updated, we can find the current run, split it if necessary, and merge with adjacent runs if needed.

But this requires O(1) access to the run for each j, which can be achieved with a dictionary.

So, the steps for updating a transition j:

1. If the transition j is being set to 0:

   a. Find the current run that contains j.

   b. Split the run into up to three parts: before j, j, and after j.

   c. Remove the original run from the runs list.

   d. Add the new runs (if any) to the runs list.

2. If the transition j is being set to 1:

   a. Check the previous transition (j-1) and next transition (j+1 mod n) to see if they are 1.

   b. Merge the current transition with any adjacent runs of 1's.

   c. Remove the adjacent runs from the runs list and add the merged run.

But this requires efficient lookups and management of the runs.

This approach is feasible but requires careful implementation.

Once the runs are maintained, the query for size s is straightforward: for each run of length L, add max(0, L - s + 2) to the answer if L >= s-1.

But how to compute this sum efficiently.

We can maintain a frequency dictionary that tracks the count of runs of each length. Then, for a query s, the answer is the sum over all L >= s-1 of (L - s + 2) * count[L].

But maintaining this frequency dictionary is possible. Each time a run is added or removed, we update the frequency counts.

But this requires O(1) time per update.

So, the steps are:

- Maintain a frequency dictionary freq where freq[L] is the number of runs of length L.

- For a query s, the answer is sum_{L >= s-1} (L - s + 2) * freq[L].

But how to compute this sum efficiently.

We can maintain a sorted list of unique run lengths. For each query s, we can iterate through the sorted list and sum the terms for L >= s-1.

But this would take O(k) time per query, where k is the number of unique run lengths. In the worst case, k is O(n), which is not feasible.

Alternatively, we can use a prefix sum array or a binary indexed tree (Fenwick tree) to track the sum for each possible L.

But since L can be up to n, this would require O(n) space, which is feasible for n up to 5e4.

So, we can maintain a Fenwick tree where each index represents a possible run length L. The Fenwick tree stores the sum of (L - s + 2) * freq[L] for each L.

But this is not directly possible, as s varies per query.

Alternatively, we can precompute for each possible L the value (L - s + 2) * freq[L] for a given s. But this is not feasible.

Hmm. This suggests that the frequency approach is not suitable.

Another idea: For each query s, the answer is the sum over all runs of max(0, L - s + 2) if L >= s-1.

This can be rewritten as sum_{L >= s-1} (L - s + 2) * freq[L].

But this is equivalent to sum_{L >= s-1} (L - (s-1) + 1) * freq[L] = sum_{L >= s-1} (L - (s-1) + 1) * freq[L].

Which is the same as sum_{L >= s-1} (L - s + 2) * freq[L].

But how to compute this sum efficiently.

We can maintain a prefix sum array for the sorted run lengths. For example, we can keep the run lengths sorted in ascending order. Then, for a given s, we can find the first run length L >= s-1 using binary search, and sum the terms from that index to the end.

But maintaining a sorted list of run lengths and a prefix sum array would require O(k) time per update, where k is the number of unique run lengths. This is not feasible for large k.

So, this approach is not suitable.

Given the time constraints, perhaps the best approach is to proceed with the segment tree approach to track the maximum consecutive 1's starting at each position, and for each query, iterate through all positions and count how many have max_consec_ones[i] >= s-1. But this is O(n) per query, which is not feasible for 5e4 queries.

But given the problem constraints, this approach would not pass the time limits.

So, what's the alternative?

Perhaps the key insight is that the maximum group length for a position i is the length of the longest possible alternating sequence starting at i. This can be represented as the maximum number of consecutive transitions that are 1, plus 1.

So, the problem reduces to maintaining the transitions array and for each query, counting the number of positions i where the maximum consecutive 1's starting at i is >= s-1.

But how to do this efficiently.

Another idea: For each position i, the maximum consecutive 1's starting at i is the same as the maximum consecutive 1's in the transitions array starting at i. So, the problem is to count the number of positions i where the maximum consecutive 1's in the transitions array starting at i is >= s-1.

This is equivalent to counting the number of positions i where there exists a run of 1's of length >= s-1 starting at i.

So, the answer is the number of positions i where the transitions array has a run of 1's of length >= s-1 starting at i.

But how to count this.

For example, a run of length L has L - (s-1) + 1 positions i where the run starts at i and has length >= s-1.

So, the total number of valid positions is the sum over all runs of max(0, L - s + 2) if L >= s-1.

This is the same as the previous approach.

So, the problem reduces to maintaining the runs of 1's in the transitions array and for each query, summing the contributions of each run.

But how to maintain the runs and their lengths efficiently.

This seems to be the crux of the problem.

So, the plan is:

- Maintain a list of runs of 1's in the transitions array.

- For each query of type 1, compute the sum of (L - s + 2) for each run of length L >= s-1.

- For each query of type 2, update the transitions array and adjust the runs accordingly.

But how to manage the runs.

Implementing this requires a data structure that can efficiently split and merge runs when transitions are updated.

In Python, this can be done using a linked list to represent the runs. Each node in the linked list contains the start and end indices of the run, and pointers to the previous and next nodes.

When a transition is updated, we can find the run that contains the transition index and split or merge as needed.

But implementing a linked list in Python is possible, but may be time-consuming.

Alternatively, we can use a list to represent the runs and manage it with binary search.

But for large n, this may not be efficient enough.

But given the time constraints, perhaps this is the best approach.

So, let's proceed with this approach.

The steps are:

1. Initialize the transitions array and compute the initial runs of 1's.

2. For each query of type 1, compute the sum of (L - s + 2) for each run of length L >= s-1.

3. For each query of type 2:

   a. Update the colors array.

   b. Update the transitions array at positions i-1 and i.

   c. For each of these positions, check if the transition has changed.

   d. For each changed transition, update the runs:

      i. If the transition was 0 and is now 1:

         - Check the previous and next transitions. If they are 1, merge the runs.

      ii. If the transition was 1 and is now 0:

         - Split the current run into up to three parts.

   e. Update the runs list accordingly.

But how to implement this.

Let's outline the code.

First, we need to compute the initial runs.

To compute the initial runs:

n = len(colors)
transitions = [0] * n
for i in range(n):
    transitions[i] = 0 if colors[i] == colors[(i+1) % n] else 1

runs = []
current
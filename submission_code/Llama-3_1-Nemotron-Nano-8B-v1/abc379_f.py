<think>
Okay, let's tackle this problem. Hmm. So the problem is about buildings arranged in a line, and for each query (l_i, r_i), we need to find the number of buildings east of r_i that are visible from both l_i and r_i. 

First, let's understand the visibility condition. A building j can be seen from building i if there's no taller building between i and j. So for building j to be visible from i, all buildings between i and j must be shorter than j. 

The task for each query is to find the count of buildings in the range r_i+1 to N that are visible from both l_i and r_i. So for each such j, it must satisfy two conditions: 
1. Between l_i and j, there's no building taller than j.
2. Between r_i and j, there's no building taller than j.

Wait, but j is east of r_i, which is east of l_i. So j is to the east of both l_i and r_i. So for j to be visible from l_i, all buildings from l_i+1 up to j-1 must be shorter than j. Similarly, for j to be visible from r_i, all buildings from r_i+1 up to j-1 must be shorter than j. 

So the j must be the maximum in the interval [l_i+1, j-1] from the west side (i.e., when viewed from l_i), and also the maximum in [r_i+1, j-1] when viewed from r_i. 

But how do we efficiently compute this for multiple queries, given the constraints of N and Q up to 2e5? We need an O(1) or O(log N) per query approach, otherwise it's going to be too slow.

Let me think about data structures. Maybe precompute for each position the next greater element to the east and west? Or perhaps for each building, track the next building that is taller than it in either direction. But how does that help with the queries?

Alternatively, for each j, can we precompute the leftmost building to the east that is taller than j? Then, for a query (l, r), the j's that are visible from l must have their leftmost taller building to the east beyond j. Similarly for visibility from r. Wait, maybe not. Let's think.

For a building j to be visible from l, all buildings between l and j must be shorter than j. So the maximum in the interval [l, j-1] must be less than j. Similarly, the maximum in [r, j-1] must be less than j.

But how do we find j's where both these conditions hold?

Hmm. Let's consider that j must be the first building to the east of r_i that is taller than all buildings between r_i and j. And also, j must be the first building to the east of l_i that is taller than all buildings between l_i and j. Wait, no. Because j could be visible from l_i even if there's a taller building further east, as long as between l_i and j there's none.

Alternatively, for j to be visible from l_i, the maximum height in the interval [l_i+1, j-1] must be less than H[j]. Similarly for r_i.

But how to compute for a given j, the maximum in the ranges [l_i+1, j-1] and [r_i+1, j-1]?

But for each query, we need to check for each j > r_i whether H[j] is larger than all in [l_i+1, j-1] and all in [r_i+1, j-1]. This seems computationally heavy if done naively for each query.

So we need a way to pre-process information that allows us to answer these range maximum queries efficiently.

But with N up to 2e5 and Q up to 2e5, even O(1) per query with O(N) pre-processing might be manageable. Wait, but for each j, we need to know for any l and r (as in the query) whether the maximum in [l+1, j-1] is less than H[j], and similarly for r.

Alternatively, for each j, precompute the leftmost position where H[j] is the maximum in some interval. Not sure.

Another approach: For a building j, the set of l's for which j is visible from l is the interval from the first building to the left of j that is taller than j. Because once there's a building taller than j to the left of j, any l to the left of that building cannot see j. Similarly, for j to be visible from r (where r is to the left of j), the r must be to the right of the last building to the left of j that is taller than j.

Wait, maybe for each j, we can compute the leftmost position where j is visible. Let's think:

The leftmost position l where j can be seen from l is the first building to the left of j that is taller than j. So l must be the nearest taller to the left. Because if there's a taller building between l and j, then j can't be seen from l. Wait, no. For example, if j is at position 5, and there's a taller building at position 3. Then, buildings at positions 1 and 2 can see j (assuming they are shorter than j). But building 3 is taller than j, so buildings to the left of 3 can't see j. So the left boundary for l is the position of the nearest taller building to the left of j. So l can be anywhere between that position and j. Wait, no. Let's see:

Suppose j is at position 5, and the nearest taller building to the left is at position 3 (H[3] > H[5]). Then, any l in [4,5] can see j. Because between l and j, there's no building taller than j. For example, l=4: the buildings between 4 and 5 are none (since j is 5). So if H[4] < H[5], then l=4 can see j. But if there's a building at 4 that is taller than j, then l=4 can't see j. Wait, but in this scenario, the nearest taller to the left is 3, which is H[3] > H[5]. Then, building 4 must be shorter than H[5], otherwise building 4 would be the nearest taller to the left of j.

So for j, the nearest taller to the left is the last building to the left of j that is taller than j. Let's call this L[j]. Similarly, the nearest taller to the right of j is R[j]. Then, for j to be visible from l, l must be >= L[j] + 1 (assuming L[j] exists). Wait, no. Because if L[j] is the last building to the left of j that is taller than j, then any l in (L[j], j) can see j, provided that between l and j there are no buildings taller than j. But since L[j] is the last such building, between L[j] and j, all buildings are shorter than j. So l can be anywhere between L[j]+1 and j. So the leftmost l that can see j is L[j]+1, and the rightmost is j-1.

Wait, maybe not. Let's take an example. Suppose j is at position 5, and L[j] is 3. So H[3] > H[5]. Then, buildings 4 and 5-1=4. So l can be 4. Because between 4 and 5, there's no building taller than j. So l=4 can see j. Similarly, l=3 can't see j, because H[3] is taller than j. So the range of l's for which j is visible from l is (L[j], j). So for l in [L[j]+1, j-1], j is visible. Also, l can be j-1, but that's the same as j's position. Wait, no, l has to be less than j. So l can be any building from L[j]+1 up to j-1, and j itself. Wait, but j is the building we are considering. So for a building j, the buildings l that can see j are those in the interval (L[j], j). Because between l and j, there's no building taller than j. Since L[j] is the last building to the left of j that is taller than j, any l to the right of L[j] (but left of j) can see j.

So for j, the visible l's are l in [L[j]+1, j-1], and l can also be j's position (but l has to be less than j, so j-1). Wait, maybe not. Because if l is j-1, then the interval between l and j is empty, so j is visible from j-1 if H[j-1] < H[j]. But in that case, L[j] would be the last building to the left of j that is taller than j. If H[j-1] is less than H[j], then L[j] would be the same as the L[j-1], but not sure.

Alternatively, maybe for each j, the left boundary is the position of the nearest taller to the left, and the right boundary is the position of the nearest taller to the right. Then, the interval where j is visible is from L[j]+1 to R[j]-1. Because between L[j] and R[j], all buildings are shorter than j. So any l in that interval can see j. Wait, but R[j] is the first building to the right of j that is taller than j. So between j and R[j], all buildings are shorter than j. So for a building k in j+1 to R[j]-1, j can see k. But for l to see j, l must be to the left of j, so that's not applicable. Hmm, maybe this line of thinking isn't directly applicable.

Alternatively, for j to be visible from l (l < j), the maximum in the interval [l+1, j-1] must be less than H[j]. So the maximum in that interval is the maximum of all H[k] where k is between l+1 and j-1. If that maximum is less than H[j], then j is visible from l.

Similarly, for j to be visible from r (r < j), the maximum in [r+1, j-1] must be less than H[j].

So for a query (l, r), we need to find all j > r such that:

max(H[l+1 ... j-1]) < H[j] and max(H[r+1 ... j-1]) < H[j]

But how to compute this efficiently for all j > r and for multiple queries?

This seems challenging. For each query, we need to check for j in r+1 to N whether both conditions hold. But with Q up to 2e5 and N up to 2e5, this would be O(Q*N), which is 4e10 operations. That's way too slow.

So we need a smarter approach.

Let me think about the possible j's that satisfy both conditions. Let's consider that for j to be visible from both l and r, j must be the maximum in the range [l, j-1] and [r, j-1]. Wait, no. Because the maximum in those ranges must be less than H[j]. So H[j] must be greater than all elements in [l+1, j-1] and in [r+1, j-1].

So H[j] is a local maximum in the ranges extending from l and r. So j must be the global maximum in the union of those ranges. Hmm.

Alternatively, j must be the maximum in the interval [l+1, j-1] and [r+1, j-1], but that's not exactly correct. Because the maximum in those ranges must be less than H[j], so H[j] is larger than all elements in both ranges.

So for j to be counted in the answer of a query (l, r), j must be the maximum in the range [l+1, j-1] and also in [r+1, j-1]. But since j is in the east of both l and r, and r is east of l, the ranges [l+1, j-1] and [r+1, j-1] may overlap.

Wait, but if r is east of l, then r+1 is at least l+1 (since l < r). So the range [r+1, j-1] is a subset of [l+1, j-1] if j-1 >= r+1. So for j to satisfy both conditions, the maximum of [l+1, j-1] must be less than H[j], and the maximum of [r+1, j-1] must also be less than H[j]. But since [r+1, j-1] is a subset of [l+1, j-1], if the maximum of the larger interval is less than H[j], then the maximum of the subset is automatically less. So the first condition (max of [l+1, j-1] < H[j]) implies the second condition (max of [r+1, j-1] < H[j]). Because [r+1, j-1] is a subset of [l+1, j-1], so if the maximum in the larger interval is less than H[j], then the subset's maximum is also less.

Wait, but that's only true if r+1 >= l+1. Which is the case because l < r. So r+1 is at least l+1 + 1? No. If l = 2 and r =3, then r+1 is 4, l+1 is 3. So the subset [4, j-1] is part of [3, j-1]. So if the maximum of [3, j-1] is less than H[j], then the maximum of [4, j-1] is also less than H[j]. So in this case, the second condition is automatically satisfied if the first is. 

So for a query (l, r), the required j's are those where j is east of r, and the maximum in [l+1, j-1] is less than H[j]. Because the second condition (max in [r+1, j-1] < H[j]) is automatically satisfied if the first condition holds.

Wait, is that correct? Let's see.

If j is east of r, then r+1 <= j-1 is possible. For example, if r is 3 and j is 5, then [r+1, j-1] is [4,4]. So if the max in [l+1,4] is less than H[5], then the max in [4,4] (which is H[4]) is also less than H[5]. But H[4] could be part of the [l+1,4] interval. So if the max of [l+1,4] is less than H[5], then H[4] is also less than H[5]. Therefore, the max of [4,4] (which is H[4]) is less than H[5]. So the second condition is automatically satisfied.

Therefore, the problem reduces to, for each query (l, r), count the number of j > r where the maximum in [l+1, j-1] is less than H[j]. 

So now the problem is to find, for each query (l, r), the number of j's in [r+1, N] such that H[j] is greater than all elements in [l+1, j-1].

How can we compute this efficiently?

The key is to pre-process for each j, the leftmost position where H[j] is the maximum from some starting point.

Alternatively, for each j, find the earliest l such that in the interval [l+1, j-1], the maximum is less than H[j]. Then, for a query (l, r), if l is <= that earliest l, then j is counted.

Wait, perhaps not. Let's think.

For j to be counted in the query (l, r), l must be such that l <= the leftmost position where the maximum in [l+1, j-1] is less than H[j]. Wait, perhaps for j, there's a range of l's where the maximum in [l+1, j-1] is less than H[j]. Then, for a query (l, r), if the query's l is within that range, then j is counted.

So for each j, we can precompute the maximum l_prev such that for any l <= l_prev, the maximum in [l+1, j-1] is less than H[j]. Then, for a query (l, r), j is counted if l <= l_prev and j > r.

But how to compute l_prev for each j?

Alternatively, for each j, find the nearest position to the left of j where H[k] >= H[j]. Let's call this position left_max[j]. Then, the maximum in [l+1, j-1] is less than H[j] if and only if l >= left_max[j] + 1. Because between left_max[j] and j, all H[k] are less than H[j]. So if l is >= left_max[j] + 1, then [l+1, j-1] is a subset of (left_max[j], j), so the maximum in that interval is less than H[j].

Wait, let's think. left_max[j] is the last position to the left of j where H[k] >= H[j]. So all positions between left_max[j]+1 and j-1 have H[k] < H[j]. So if l is >= left_max[j] + 1, then l+1 is >= left_max[j]+2, so the interval [l+1, j-1] is within the range of H[k] < H[j]. Therefore, the maximum in that interval is less than H[j]. 

So for j to be counted in the query (l, r), the query's l must be <= left_max[j] (but no, because l is the left end of the query, and j is to the east of r which is east of l). Wait, no. The query is (l_i, r_i), and j must be east of r_i. So for j to be considered, j must be > r_i. Also, for j to be visible from l_i, l_i must be such that the maximum in [l_i+1, j-1] is < H[j]. Which requires that l_i+1 > left_max[j], i.e., l_i >= left_max[j]. Because if left_max[j] is the last position where H[k] >= H[j], then any l_i that is >= left_max[j] would have l_i+1 > left_max[j], so the interval [l_i+1, j-1] is within the range where all H[k] < H[j].

Wait, perhaps I got the logic reversed. Let's rephrase:

left_max[j] is the last position to the left of j where H[k] >= H[j]. So between left_max[j] and j, all H[k] are < H[j]. So for any l_i such that l_i >= left_max[j], the interval [l_i+1, j-1] is a subset of (left_max[j], j), so all H[k] in that interval are < H[j]. Therefore, the maximum in that interval is < H[j], so j is visible from l_i.

So for j to be visible from l_i, l_i must be >= left_max[j]. Because if l_i is less than left_max[j], then the interval [l_i+1, j-1] would include left_max[j], which has H[k] >= H[j], making the maximum >= H[j], so j is not visible from l_i.

Therefore, the condition for j to be visible from l_i is l_i >= left_max[j].

So for a query (l, r), we need to count the number of j > r where l >= left_max[j], and j > r.

Wait, no. Because l is the l_i in the query. So for j to be visible from l, l must be >= left_max[j]. And j must be > r.

So for each query (l, r), the answer is the number of j in [r+1, N] where left_max[j] <= l.

So the problem reduces to, for each query (l, r), count the number of j in [r+1, N] such that left_max[j] <= l.

If we can precompute for each j the left_max[j], then for each query, we can find the count of j in [r+1, N] where left_max[j] <= l.

How to compute this efficiently?

We can precompute an array left_max, where left_max[j] is the index of the last building to the left of j with H[k] >= H[j]. Then, for each j, left_max[j] is the position of the closest greater or equal element to the left.

Once we have left_max for all j, the problem becomes, for each query (l, r), count the number of j in [r+1, N] where left_max[j] <= l.

This is a range query problem. We can process this by building a data structure that allows us to query, for a range [a, b], how many elements have left_max[j] <= l.

But with N up to 2e5 and Q up to 2e5, we need a way to answer these queries in O(log N) time per query.

A possible approach is to pre-process for each j, store left_max[j], and then build a segment tree or a binary indexed tree (Fenwick tree) that can answer the count of elements in a range [r+1, N] with left_max[j] <= l.

But since the queries are for ranges starting from r+1 to N, and varying l, perhaps we can pre-process the left_max array and build a structure that allows for each position, to query the count of elements in [a, b] with left_max[j] <= x.

But how?

An alternative approach is to process the queries offline. We can sort the queries by their r in increasing order, and process them in that order, maintaining a data structure that allows us to query the number of elements j >= current r (but since r is increasing, we can process j from N down to 1, adding them to the data structure when r passes their position). Wait, maybe not. Let's think.

Alternatively, for each j, left_max[j] is known. We can create an array where for each j, we have a tuple (j, left_max[j]). Then, for each query (l, r), we need to count the number of j in (r, N] (i.e., j > r) where left_max[j] <= l.

So this is equivalent to a 2D range query: count the number of j where j > r and left_max[j] <= l.

To handle this, we can process the queries offline. Sort the queries in increasing order of r. Then, we can process j from N down to 1, and for each j, we insert it into a data structure that allows us to query how many elements have left_max[j] <= l. 

Here's the idea:

- We sort all queries by their r in increasing order. So we process queries with smaller r first.

- We process j from N down to 1. For each j, when we reach j = current r (from the query), we can start adding j to the data structure. Wait, no. Because the queries are sorted by r, and for each query (l, r), we need to consider j > r. So for a query with r = x, all j > x must be in the data structure.

So, as we process j in decreasing order (from N to 1), whenever j is greater than the current query's r, we can add j to the data structure. Then, when all j > r are in the data structure, we can query how many have left_max[j] <= l.

Wait, perhaps not. Let's see. Suppose we have a list of queries sorted by r in increasing order. For example, the first query has r=2, the next has r=5, etc. 

We process j from N down to 1. For each j, when j is processed, we add it to the data structure. Then, for all queries with r < j, when their r is less than j, j is in the data structure. Wait, this might not work. Let's think:

If we process queries in order of increasing r, then for each query (l, r), we need to include all j > r. So for the first query (r=2), we need to include j=3,4,5,...N. So if we process j from N down to 1, and for each j, after adding it to the data structure, we check if there are any queries with r < j. But this seems tricky.

Alternatively, the standard approach for offline processing of such queries is to sort the queries in increasing order of r. Then, for each j from N down to 1, we add j to the data structure when j > r (the current query's r). Wait, but how to manage this?

Alternatively, here's a possible approach:

- Sort all queries in increasing order of r.

- Initialize a pointer to the first query. For each j starting from N down to 1:

   - While the current query's r is less than j, process the query by querying the data structure for the count of elements with left_max[j] <= l. Then, move to the next query.

   - Add j to the data structure (i.e., insert left_max[j] into the data structure).

But this way, for j=N, we process all queries with r < N. Then j=N-1, process queries with r < N-1 but >= N-1 (but since queries are sorted, once r >= j, we stop processing). Hmm, maybe not. Let's see.

Wait, perhaps the correct approach is:

We process j from N down to 1. For each j, we first add j to the data structure. Then, we process all queries with r equal to j-1. Because when j is added to the data structure, it's available for all queries where r < j. So for example, when j=5, adding it to the data structure allows all queries with r <5 to include j in their count.

But how to manage this? Let's think of the queries sorted by r. So for each j, after adding it to the data structure, we process all queries that have r = j-1. Because those queries have r = j-1, and j is now in the data structure (since j > r), so j can be considered for those queries.

Wait, but queries can have r values anywhere. So perhaps the correct approach is:

Sort the queries in increasing order of r. Then, for each query in this order, we need to include all j > r in the data structure. So when processing a query with r = x, all j > x must be in the data structure. 

To achieve this, we can process the queries in order of increasing r. We can process j from N down to 1. For each j, when j is processed (added to the data structure), we check if there are any queries with r < j and have not been processed yet. For those queries, since j is now in the data structure, they can include j in their count.

But this is a bit vague. Let's think of an example. Suppose we have queries sorted by r as (r=2), (r=3), (r=5). 

We process j from N down to 1. For j=N, add to data structure. Then, check if any queries have r < j (which is N). All queries with r < N are processed. But the first query is r=2, which is less than N. So we process that query now, using the current data structure (which includes j=N). Then, process the next query (r=3) if it's r < j (but j is N, so yes). But this approach would process all queries in order, but j is N, which is larger than all r's. So this is not correct.

Alternatively, perhaps the correct way is to process queries in order of increasing r, and for each query (l, r), we need to include all j > r. So when processing the query (l, r), all j > r must have been added to the data structure. Since queries are sorted by r, and we process them in order, we can process j from N down to 1, and for each j, after adding it to the data structure, we check if there are any queries with r < j. For those queries, j is now in the data structure and can be considered.

But how to track which queries have r < j and haven't been processed yet. 

Alternatively, here's a step-by-step plan:

1. Precompute left_max for all j.

2. Sort all queries in increasing order of r. Let's store them as a list sorted by r.

3. Initialize a Fenwick Tree (or Binary Indexed Tree) or a Segment Tree that can efficiently count the number of elements in the structure with left_max[j] <= l. The elements in the structure are the j's that have been processed (i.e., added to the data structure).

4. Initialize a pointer i to 0 (points to the current query to process).

5. Iterate j from N down to 1:

   a. Insert left_max[j] into the Fenwick Tree. (But how? Because for each j, we need to store left_max[j] in the data structure. The Fenwick Tree can be used to track the frequency of left_max[j] values. Then, for a query l, the number of elements with left_max[j] <= l is the sum of frequencies up to l.)

   b. While the i-th query (sorted by r) has r < j:

      i. For the current query (l, r), the answer is the number of elements in the Fenwick Tree with left_max[j] <= l and j > r. Since we have sorted the queries by r and process j in decreasing order, all j > r have already been added to the Fenwick Tree. So querying the Fenwick Tree for the count of elements with left_max[j] <= l gives the required answer.

      ii. Record the answer for this query.

      iii. Move to the next query (i += 1).

This way, as we process j from N down to 1, we add j to the Fenwick Tree. Then, for each query (l, r) where r is less than j, we process the query. Since the queries are sorted by r, once we process a query with r = x, all j > x are already in the Fenwick Tree. So for that query, the count is the number of elements in the Fenwick Tree with left_max[j] <= l.

This approach should work. Let's see with the sample input.

Sample Input 1:

5 3

H = [2, 1,4,3,5]

left_max for each j:

j=1: no element to the left. left_max[1] = 0 (assuming 0 represents no such element).

j=2: look left. H[1]=2 > H[2]=1. So left_max[2] =1.

j=3: look left. H[1]=2 < H[3]=4. H[2]=1 <4. So no element >=4. left_max[3]=0.

j=4: look left. H[3]=4 >3. So left_max[4]=3.

j=5: look left. H[3]=4 <5, H[4]=3 <5. So left_max[5]=0.

So left_max array is [0,1,0,3,0].

Queries sorted by r:

First query: (1,2) r=2.

Second query: (3,5) r=5.

Third query: (1,4) r=4.

Processing j from 5 down to 1:

j=5: add left_max[5]=0 to Fenwick Tree. Now, check if any query has r <5. The first query has r=2 <5. The second query has r=5, which is not <5. So process the first query (l=1, r=2). The Fenwick Tree contains j=5. left_max[j]=0 <=1. So count is 1. But wait, the query is (1,2), and j must be >2. j=5 is >2. So the answer for the first query is 1. But according to the sample output, the answer is 2. Hmm, this suggests a problem.

Wait, the sample input's first query is l=1, r=2. The answer is 2. According to our approach, j=5 is added, and the query (1,2) is processed. The Fenwick Tree has j=5 (left_max=0 <=1). So count is 1. But the correct answer is 2. So where's the mistake?

Ah, because when j=5 is added, we also have j=3,4, etc. Wait, but we process j from 5 down to 1. So j=5 is added first. Then j=4 is added next. Then j=3, etc.

Wait, the steps would be:

Process j=5:

Add left_max[5]=0 to Fenwick Tree.

Check if any query has r <5. The first query has r=2. So process it. The Fenwick Tree has j=5. left_max[j] =0 <=1. So count is 1. But according to the sample, the answer is 2. So this approach is missing j=3 and j=5. Wait, what's the correct j's for the first query.

The first query is l=1, r=2. The j's must be >2. So j=3,4,5.

For j=3: left_max[j] =0 <=1. So yes. So j=3 is counted.

For j=4: left_max[j]=3. 3>1. So no.

For j=5: left_max[j]=0 <=1. So yes.

So the correct count is 2 (j=3 and 5). But according to the approach, when j=5 is processed, the Fenwick Tree contains only j=5. Then, the query (1,2) is processed, and the count is 1. But j=3 and j=5 are both valid. So why isn't j=3 in the Fenwick Tree?

Because j=3 is processed after j=5. Because we process j in decreasing order. So j=5 is added first, then j=4, then j=3, etc.

So when j=5 is added, the Fenwick Tree has j=5. Then, the query (1,2) is processed. But j=3 hasn't been added yet. So the count is 1. But j=3 is also a valid j, but it's not in the data structure yet.

So the approach is incorrect. Because when processing queries in order of increasing r, and j is processed in decreasing order, some j's may not have been added to the data structure when the query is processed.

Ah, this is the problem. The approach processes j from N down to 1. So when processing a query with r=2, j=5 is added (since 5>2), but j=3 is also >2. However, j=3 is processed after j=5. So when the query is processed, j=3 hasn't been added to the data structure yet. So the approach misses j=3.

So the order of processing j is incorrect. We need to process j in increasing order of r. Wait, no. Because the queries are sorted by r in increasing order. So for a query with r=2, we need all j>2 to be in the data structure. But j=3,4,5 are all >2. So when processing the query (1,2), all these j's should be in the data structure.

But if we process j from N down to 1, then j=5 is added first, then 4, then 3, then 2, then 1. So when the query (r=2) is processed, j=3,4,5 are already in the data structure. Because j=5 is added first, then 4, then 3. So when the query (r=2) is processed, the Fenwick Tree contains j=5,4,3. Because j=3 is processed after j=5 and 4. But when is the query processed?

Let's re-examine the algorithm:

The algorithm processes j from N down to 1. For each j, after adding it to the Fenwick Tree, it checks if there are any queries with r < j. For each such query, it processes them.

So for j=5:

Add to Fenwick Tree.

Check if any queries have r <5. Yes, the first query (r=2). Process it. The Fenwick Tree contains j=5. So count is 1.

Then, j=4 is processed. Add to Fenwick Tree. Check if any queries have r <4. The first query has r=2 <4. So process it again. But wait, the first query was already processed. So this is a problem. The algorithm would process the same query multiple times.

Ah, right. Because the queries are sorted by r, and for j=5, r=2 is <5, so the query is processed. Then j=4, r=2 is also <4, so the query is processed again. This leads to incorrect results.

So the algorithm as described is flawed. The problem is that the same query can be processed multiple times as j decreases.

This suggests that the approach of processing j in decreasing order and queries in increasing order of r is incorrect.

Alternative approach: process the queries in increasing order of r, and for each query, add all j > r to the data structure before processing the query. But how to do this efficiently.

Another idea is to process the queries in order of increasing r. For each query (l, r), we need to include all j > r. So, we can process the j's in increasing order, and for each j, when j is processed, we add it to the data structure. Then, when the current query's r is less than j, we can process the query.

But this seems difficult. Alternatively, we can process j from 1 to N. For each j, after adding it to the data structure, process all queries with r = j-1. Because j is now in the data structure for all queries with r < j.

But this requires that the queries are sorted by r. So:

1. Sort the queries in increasing order of r.

2. For each query, store its index so that we can output the answers in the correct order.

3. Iterate j from 1 to N:

   a. Add j to the data structure (insert left_max[j]).

   b. Process all queries with r = j-1. For each such query, query the data structure for the count of elements with left_max[j] <= l.

But wait, j is the current position. When j is processed, it's added to the data structure. But queries with r = j-1 require j > r, which is j > j-1. So j is now in the data structure, but j must be > r = j-1. So j is indeed > r.

But for example, if j=3, then r=2. So queries with r=2 will include j=3,4,5,...N. But j=3 is added now, but j=4,5, etc. are added later. So this approach is incorrect.

Hmm. So this approach would only include j up to the current j. So for a query (l, r= j-1), j is added to the data structure, but higher j's are not. So this approach is not correct.

So this suggests that the correct approach is to process the queries in increasing order of r, and for each query, add all j > r to the data structure. But how to do this efficiently.

An alternative idea is to use a segment tree where each node represents a range of j's. For each query (l, r), we need to query the segment [r+1, N] and count the number of j's in that segment where left_max[j] <= l.

This can be achieved with a segment tree that allows range queries for the count of elements with left_max[j] <= l in a given interval.

But building such a segment tree requires that each node can answer the query of how many elements in its interval have left_max[j] <= l. This can be done by storing the elements in each node's interval in a sorted list. Then, for a query l, we can perform a binary search in each relevant node's list to find the count.

The time complexity for each query would be O(log N * log K), where K is the maximum size of a node's interval. Since the segment tree has O(log N) levels, and each node's list is sorted, this is manageable for Q=2e5.

The construction time for the segment tree is O(N log N), as each element is inserted into O(log N) nodes, and each insertion is O(1) if we pre-sort.

So here's the plan:

1. Precompute left_max for all j.

2. Build a segment tree where each node represents a range of j's. For each node, store the left_max[j] values in a sorted list.

3. For each query (l, r), perform a range query on the segment [r+1, N] to count the number of j's in this range where left_max[j] <= l.

This approach would allow each query to be answered in O(log N * log N) time (since each level of the segment tree requires a binary search on a sorted list of O(N) elements). For Q=2e5, this would be 2e5 * (20)^2 = 8e6 operations, which is acceptable.

But building the segment tree with sorted lists may require O(N log N) time and O(N log N) space, which is acceptable for N=2e5.

So the steps are:

- Precompute left_max for each j.

- Build a segment tree where each node covers a range of j's and stores the left_max[j] values in a sorted list.

- For each query (l, r), query the segment [r+1, N] and count the number of elements with left_max[j] <= l.

Now, the key is to implement the segment tree efficiently.

Let's think about the segment tree structure. Each node represents a range [start, end]. For a leaf node (start == end), the sorted list contains just left_max[start]. For internal nodes, the sorted list is the merge of the sorted lists of the left and right children.

Building the segment tree recursively:

function build(node, start, end):
    if start == end:
        node's list = [left_max[start]]
    else:
        mid = (start + end) // 2
        build left child (start, mid)
        build right child (mid+1, end)
        node's list = merge left child's list and right child's list, sorted.

This way, each node's list is a sorted list of left_max[j] for j in its range.

The merge step takes O(a + b) time for two sorted lists of size a and b. For a segment tree, the total time to build is O(N log N), since each element is part of O(log N) nodes, and each merge step for a node of size k takes O(k) time. The total time is O(N log N).

Once the segment tree is built, to query a range [a, b], we recursively traverse the tree, visiting nodes that overlap with [a, b]. For each such node, we perform a binary search on its sorted list to count the number of elements <= l. Summing these counts gives the total.

So the query function would look like:

function query(node, node_start, node_end, a, b, l):
    if node_end < a or node_start > b:
        return 0
    if a <= node_start and node_end <= b:
        # count elements <= l in node's list
        return bisect_right(node.list, l)
    mid = (node_start + node_end) // 2
    left_count = query(left_child, node_start, mid, a, b, l)
    right_count = query(right_child, mid+1, node_end, a, b, l)
    return left_count + right_count

This way, each query takes O(log N * log K) time, where K is the maximum size of a node's list. Since each node's list is sorted, the binary search is O(log K), and K is up to O(N) for the root node, but in practice, for a segment tree, each element is in O(log N) nodes, so the average K per node is O(N / log N), but the worst case for a query would be O(log^2 N) time.

For N=2e5, log2(N) is around 18, so each query would take around 18 * 18 = 324 operations, which is acceptable for 2e5 queries.

Now, the next step is to implement this in Python.

But in Python, implementing a segment tree with merged sorted lists may be challenging due to time and memory constraints. Let's see:

For N=2e5, the segment tree will have O(4*N) nodes. Each node's list can be up to O(N) in size (the root node's list is all elements). So the total memory would be O(N log N), which for N=2e5 is 2e5 * 20 = 4e6 elements. Which is manageable.

But merging sorted lists for each node can be time-consuming. For example, the root node's list is the merged list of left and right children. Each of which is a merged list of their children, and so on. The total time to build the segment tree would be O(N log N), but in practice, for Python, this might be tight for N=2e5.

Alternative approach: precompute for each j, the left_max[j], then build a binary indexed tree (Fenwick Tree) for each possible l. But that's not feasible.

Another alternative is to use a wavelet tree or a range tree. But implementing those in Python may be challenging.

Alternatively, we can use a sorted list for each j and use binary search. But I think the segment tree approach is the way to go.

Now, let's proceed to implement the segment tree.

But wait, in Python, recursion depth may be a problem. For N=2e5, the recursion depth can be up to log2(2e5) ~ 18, which is acceptable.

So the steps are:

1. Read N and Q.

2. Read H array.

3. Precompute left_max for each j.

   To compute left_max[j], we can use a stack. For each j, left_max[j] is the index of the last element to the left of j that is >= H[j]. This is a standard problem, similar to the Next Greater Element (NGE) problem.

   The algorithm for left_max:

   Initialize an empty stack. For each j from 0 to N-1 (assuming 0-based index):

      while stack is not empty and H[stack[-1]] < H[j]:
          stack.pop()
      if stack is empty:
          left_max[j] = 0
      else:
          left_max[j] = stack[-1] + 1  # assuming 1-based indices
      stack.append(j)

   Wait, but the problem uses 1-based indices. So for j=1 to N:

   left_max[j] is the largest index i < j where H[i] >= H[j]. If no such i exists, left_max[j] =0.

   So for example, for j=1, there's no i <1, so left_max[1] =0.

   So the code for left_max:

   stack = []
   left_max = [0]*(N+1)  # 1-based
   for j in range(1, N+1):
       while stack and H[stack[-1]] < H[j]:
           stack.pop()
       if stack:
           left_max[j] = stack[-1]
       else:
           left_max[j] = 0
       stack.append(j)

   Wait, but H is 1-based or 0-based? The input is given as H_1 H_2 ... H_N. So in code, it's better to read H as 1-based.

   So for example, H[1], H[2], ..., H[N].

   So the code for left_max would be:

   stack = []
   left_max = [0]*(N+2)  # indices 1..N
   for j in range(1, N+1):
       while stack and H[stack[-1]] < H[j]:
           stack.pop()
       if stack:
           left_max[j] = stack[-1]
       else:
           left_max[j] = 0
       stack.append(j)

   This correctly computes the last greater or equal element to the left of j. Wait, no. Because the condition is H[stack[-1]] >= H[j]. But in the code above, we pop elements from the stack while H[stack[-1]] < H[j]. So the remaining stack elements (if any) have H >= H[j]. So the top of the stack is the last element >= H[j]. So the code is correct.

   For example, j=3 in the sample input:

   H[3] =4. The stack before processing j=3 is [1,2]. H[1]=2 <4, H[2]=1 <4. So pop both. Stack is empty. left_max[3] =0.

   For j=4, H[4]=3. Stack is [3]. H[3] =4 >=3. So left_max[4] =3.

   For j=5, H[5]=5. Stack is [3,4]. H[3]=4 <5, H[4]=3 <5. So pop both. Stack is empty. left_max[5]=0.

   Which matches the sample.

4. Build the segment tree with each node's list being the sorted list of left_max[j] in its range.

5. For each query (l, r), the answer is the number of j in [r+1, N] where left_max[j] <= l.

   So for each query, we perform a range query on the segment tree for the interval [r+1, N], and count the number of elements <= l.

Now, implementing the segment tree in Python.

But implementing a segment tree with merged sorted lists for each node can be done with a class-based approach. However, in Python, for large N, this may be slow. But given the time constraints, we can proceed.

Alternatively, we can represent each node's list as a list, and build the segment tree recursively.

Here's a possible implementation:

class SegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.size = 1
        while self.size < self.n:
            self.size <<=1
        self.tree = [[] for _ in range(2*self.size)]
        # Build the leaves
        for i in range(self.n):
            self.tree[self.size + i] = [data[i]]
        # Build the internal nodes
        for i in range(self.size-1, 0, -1):
            self.tree[i] = sorted(self.tree[2*i] + self.tree[2*i+1])
    
    def query(self, l, r, x):
        # Query [l, r] 0-based, count elements <=x
        res =0
        l += self.size
        r += self.size
        while l <= r:
            if l %2 ==1:
                res += bisect.bisect_right(self.tree[l], x)
                l +=1
            if r %2 ==0:
                res += bisect.bisect_right(self.tree[r], x)
                r -=1
            l >>=1
            r >>=1
        return res

But this is a binary heap-based segment tree. However, the data is 1-based or 0-based? Need to adjust the indices.

But for the given problem, the data is 1-based (j ranges from 1 to N). So the data array passed to the segment tree is left_max[1..N], which is 1-based. So in the segment tree, the leaves are 0-based. So need to adjust the indices accordingly.

Alternatively, we can treat the data as 0-based. For example, j ranges from 1 to N, but in the segment tree, the data is stored from index 0 to N-1.

So for the sample input, data would be [0,1,0,3,0] (left_max[1..5] = [0,1,0,3,0]).

But when building the segment tree, the data is passed as a list from index 1 to N. So perhaps the code should be adjusted to handle 1-based indices.

Alternatively, in the code, the data is stored as a list from index 0 to N-1, where data[i] corresponds to j=i+1.

But this can get confusing. To avoid confusion, let's adjust the code to 0-based indices.

So for j in 1..N, we have data[j-1] = left_max[j].

So the segment tree is built on data[0..N-1], which corresponds to j=1..N.

Then, for a query (l, r), the j's are from r+1 to N. So in 0-based terms, j starts at r (since j=r+1 is data[r] in 0-based).

So the query range is [r, N-1].

For example, in the first sample query (l=1, r=2), the j's are 3,4,5. In 0-based data indices, j=3 is data[2], j=4 is data[3], j=5 is data[4]. So the query range is [2,4].

So the code for the segment tree's query function should take l and r in 0-based indices.

But the current segment tree code uses 0-based indices for the data array.

So the steps are:

- Read N and Q.

- Read H as a list of N elements (1-based).

- Compute left_max for j=1..N, stored in a list data of length N, where data[j-1] is left_max[j].

- Build the segment tree with this data.

- For each query (l_i, r_i), compute the answer as the number of j in [r_i+1, N], which in 0-based data indices is [r_i, N-1], where left_max[j] <= l_i.

- So the query to the segment tree is the range [r_i, N-1], and we count the number of elements <= l_i.

Now, the segment tree code needs to handle these 0-based indices.

The segment tree code provided earlier uses a binary heap-like structure, which may not be the most efficient for Python. But let's proceed.

Testing with the sample input:

Sample Input 1:

5 3

H = [2,1,4,3,5]

left_max for j=1..5:

j=1: 0 → data[0] =0

j=2:1 → data[1] =1

j=3:0 → data[2] =0

j=4:3 → data[3] =3

j=5:0 → data[4] =0

So data = [0,1,0,3,0]

Building the segment tree:

The leaves are [0], [1], [0], [3], [0].

Internal nodes are built by merging. For example, the parent of leaves 0 and 1 (indices 5 and 6 in the binary heap-based tree) would have [0,1]. The parent of leaves 2 and 3 (indices 7 and 8) would have [0,3]. The parent of leaves 4 and (none) would have [0]. The root node would merge all leaves.

But this is a bit complicated. However, when querying the range [2,4] (j=3,4,5), which corresponds to data indices 2,3,4 (values 0,3,0), the segment tree would return the count of elements <= l_i.

For the first query, l_i=1, r_i=2. So the query range is [2,4], and l_i=1.

The data in this range is [0,3,0]. The elements <=1 are 0,0. So count is 2. Which matches the sample.

So the code would correctly return 2 for the first query.

The other queries can be checked similarly.

Now, the problem is to implement this in Python, considering the time and memory constraints.

But for N=2e5, the segment tree would have O(4*N) nodes, and each node's list is a sorted list. The total memory is O(N log N), which for N=2e5 is about 2e5 * 20 =4e6 elements. Which is manageable.

However, building the segment tree in Python with the current approach may be too slow. Because merging two lists for each node is O(k), and for N=2e5, the time could be O(N log N), which is 2e5 * 20 =4e6 operations. But in Python, this may be acceptable.

But merging two lists of size a and b takes O(a + b) time. For each level of the segment tree, the total time is O(N). Since there are log N levels, the total time is O(N log N), which is acceptable.

But for N=2e5, log N is 18, so 2e5 * 18 =3.6e6 operations, which should be manageable.

Now, let's write the code.

First, read N and Q.

Then, read H as a list. Compute left_max.

Then, build the segment tree.

Then, process each query.

But wait, the queries are given as l_i and r_i, which are 1-based. So for a query (l_i, r_i), the j's are in [r_i+1, N], which in data indices is [r_i, N-1].

So the code for the query is:

for each query l, r:

   ans = st.query(r, N-1, l)

But wait, in the segment tree, the data is 0-based. So for j from r_i+1 to N, the data indices are r_i to N-1.

Yes.

Now, let's implement the segment tree.

But the previous code for the segment tree uses a binary heap-like structure. However, for large N, this may not be the most efficient. Another approach is to build the segment tree recursively.

Alternatively, use a list of lists to represent the tree.

But in Python, for large N, the recursive approach may hit stack limits, but for N=2e5, it's unlikely.

Here's an alternative implementation of the segment tree using a recursive approach:

def build_tree(data, l, r, node):
    if l == r:
        tree[node] = [data[l]]
        return
    mid = (l + r) // 2
    build_tree(data, l, mid, 2*node)
    build_tree(data, mid+1, r, 2*node+1)
    tree[node] = sorted(tree[2*node] + tree[2*node+1])

But this requires pre-defining the size of the tree. For N=2e5, the size of the tree can be up to 4*N.

But in Python, lists are dynamic, but this approach may be too slow for large N.

Alternatively, we can use a list of lists and build the tree in a bottom-up manner.

But perhaps the fastest way in Python is to use a library like bisect and build the segment tree with each node's list sorted.

But given time constraints, let's proceed with the code.

Putting it all together:

import bisect

n, q = map(int, input().split())
H = list(map(int, input().split()))
H = [0] + H  # Make it 1-based

left_max = [0]*(n+2)  # 1-based
stack = []
for j in range(1, n+1):
    while stack and H[stack[-1]] < H[j]:
        stack.pop()
    if stack:
        left_max[j] = stack[-1]
    else:
        left_max[j] = 0
    stack.append(j)

# Build segment tree on left_max[1..n], which is data[0..n-1] (0-based)
data = left_max[1:n+1]

# Implement a segment tree here
class SegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.size = 1
        while self.size < self.n:
            self.size <<=1
        self.tree = [[] for _ in range(2*self.size)]
        # Fill leaves
        for i in range(self.n):
            self.tree[self.size + i] = [data[i]]
        # Build parents
        for i in range(self.size-1, 0, -1):
            self.tree[i] = sorted(self.tree[2*i] + self.tree[2*i+1])
    
    def query_range(self, l, r, x):
        # l and r are 0-based indices in the data array
        res = 0
        l += self.size
        r += self.size
        while l <= r:
            if l % 2 == 1:
                res += bisect.bisect_right(self.tree[l], x)
                l +=1
            if r %2 ==0:
                res += bisect.bisect_right(self.tree[r], x)
                r -=1
            l >>=1
            r >>=1
        return res

st = SegmentTree(data)

for _ in range(q):
    l, r = map(int, input().split())
    # j must be > r, so data indices are r to n-1
    a = r
    b = n-1
    ans = st.query_range(a, b, l)
    print(ans)

But this code may have issues with the segment tree construction for large N. For example, when N is 2e5, the segment tree's size is 2^18 = 262144 (since 2^17 is 131072, which is less than 2e5). So the size is sufficient.

But merging the lists for each node may take a lot of time. For example, merging two lists of size 1e5 each would take 2e5 operations. For each level of the tree, this is O(N) time. So for log N levels, it's O(N log N) time, which is acceptable.

But in Python, this may be too slow for N=2e5. For example, merging two lists of 1e5 elements each would take 2e5 operations per node, and there are O(N) such nodes. So for N=2e5, this would be O(2e5 * 2e5) = 4e10 operations, which is way too slow.

So this approach is not feasible.

Alternative approach: use a different data structure to answer the range maximum queries.

Wait, but the original problem requires counting the number of j's in a range where left_max[j] <= l. This is equivalent to a range query for the count of elements <= l in a certain range.

This is a standard problem that can be solved with a Wavelet Tree or a range-moment query structure, but these are not standard in Python.

Alternative idea: precompute for each j, left_max[j], and then for each j, store the left_max[j] in an array. Then, for each query (l, r), we need to count the number of elements in the range [r+1, N] where left_max[j] <= l.

This can be done with a prefix array for each possible l, but that's not feasible.

Another approach: for each j, left_max[j] can be up to j-1. So for each j, we can store (left_max[j], j) and sort them. Then, for a query (l, r), we need to find the number of j's > r where left_max[j] <= l.

This can be done with a binary search.

But how?

We can pre-process all j's and sort them by j. For each j, store left_max[j]. Then, for a query (l, r), we need to count the number of j's in (r, N] where left_max[j] <= l.

This can be represented as a 2D problem: for each j, we have (j, left_max[j]). For a query (l, r), count the number of j's where j > r and left_max[j] <= l.

This is equivalent to a range query in a 2D plane: j ranges from r+1 to N, and left_max[j] ranges from 0 to l.

To handle this, we can use a binary indexed tree (Fenwick Tree) with events sorted by j.

Here's the idea:

1. Sort all j's in increasing order (from N down to 1).

2. Sort the queries in increasing order of r.

3. For each query (l, r), we need to process all j's > r and left_max[j] <= l.

But how to process this efficiently.

Alternatively, we can process the queries in increasing order of r. For each j from N down to 1, we insert j into a data structure that allows us to query the number of elements with left_max[j] <= l. 

This is similar to the initial approach but with a Fenwick Tree.

So here's the plan:

- Precompute left_max for all j.

- Sort the queries in increasing order of r, keeping track of their original indices.

- Process j from N down to 1:

   a. Insert left_max[j] into a Fenwick Tree (or another structure) that can count the number of elements <= l.

   b. For all queries with r < j, process them by querying the Fenwick Tree for the count of elements <= l.

But how to manage the queries.

This is the same as the initial approach, but using a Fenwick Tree instead of a segment tree. 

The Fenwick Tree approach requires that the left_max[j] can be compressed into a range of integers to fit the BIT's indices.

So steps:

1. Compress the left_max values. Since left_max[j] can be up to j-1 (which is up to N), but H is unique (given H_i != H_j for i !=j), but left_max[j] can be any previous index. So left_max[j] can be up to N. So the possible values of left_max[j] are 0,1,2,...,N.

But N is up to 2e5, which is manageable for a Fenwick Tree.

So no need for coordinate compression.

2. Initialize a Fenwick Tree that can count the number of elements <= x.

But a Fenwick Tree can be used to count the number of elements <= x by maintaining a frequency array and using prefix sums.

So each time we insert a left_max[j], we update the frequency array by adding 1 at position left_max[j]. Then, querying the number of elements <= x is a prefix sum up to x.

But the Fenwick Tree can do this in O(log M) time, where M is the maximum possible value of left_max[j] (which is N).

So the steps are:

- Sort the queries in increasing order of r, and store them in a list with their original indices.

- Process j from N down to 1:

   a. Insert left_max[j] into the Fenwick Tree (add 1 at position left_max[j]).

   b. For all queries with r < j, process them by querying the Fenwick Tree for the sum up to l, and store the result.

But how to efficiently process the queries when j is processed.

Here's the correct approach:

- Sort the queries in increasing order of r.

- Initialize a pointer i to 0.

- Iterate j from N down to 1:

   a. Insert left_max[j] into the Fenwick Tree.

   b. While i < Q and queries[i].r < j:

      i. The current query has r = queries[i].r.

      ii. The answer for this query is the Fenwick Tree's prefix sum up to l.

      iii. Store the answer.

      iv. i += 1.

This way, for each query (l, r), when j is processed (from N down to 1), we check if j is greater than r. If so, the Fenwick Tree contains all j's > r. So when the query's r is less than j, we process the query.

But since the queries are sorted by r in increasing order, once we process a query with r = x, all j > x have been added to the Fenwick Tree. So for example, for the first query with r=2, when j=5 is processed, we check if r=2 <5. Yes. So process the query. Then j=4 is processed, check if r=2 <4. Yes. Process the query again. But this would lead to processing the same query multiple times.

Ah, this is the same problem as before. So the initial approach is incorrect.

The correct way is to process each query exactly once when the j reaches r+1. But this requires a different approach.

Alternative approach:

- Sort the queries in increasing order of r.

- For each query, we need to include all j > r. So when j is processed from N down to 1, once j becomes <= r, we can stop processing that query.

But how to track which queries have r >= j.

So here's the correct algorithm:

Sort the queries in increasing order of r.

Initialize a pointer i=0.

Process j from N down to 1:

   Insert left_max[j] into the Fenwick Tree.

   while i < Q and queries[i].r < j:

       query the Fenwick Tree for sum <= l.

       store the answer.

       i +=1

This way, for each j, we process all queries with r < j. But since queries are sorted by r, once we process a query with r = x, all subsequent queries will have r >=x. So for example, if j=5 is processed, all queries with r<5 are processed. Then, j=4 is processed, and queries with r<4 are processed. But since the queries are sorted by r, once we processed all queries with r<5, the next queries have r>=5. So when j=4, there are no queries with r<4, so the while loop doesn't process anything.

This way, each query is processed exactly once when j is the first j that is greater than r. For example, a query with r=2 will be processed when j=3,4,5, etc. But this would process the query multiple times, which is incorrect.

So this approach is incorrect.

The correct approach is to process each query once when j reaches r+1. But how?

The correct way is to process the queries in the order of increasing r. For each query (l, r), we need to process it after all j > r have been added to the Fenwick Tree.

So the algorithm should be:

Sort the queries in increasing order of r.

Initialize a pointer i=0.

Process j from N down to 1:

   insert left_max[j] into the Fenwick Tree.

   while i < Q and queries[i].r < j:

       query the Fenwick Tree for sum <= l.

       store the answer.

       i +=1

But this would process a query (l, r) when j is the first j > r. For example, for r=2, j=3 is processed, and the query is processed. But j=4,5 are also >2, but the query is processed only once. Because once i increments past the query, it's not processed again.

Wait, no. Because the queries are sorted by r. So once we process a query with r=2, all subsequent queries have r >=2. When j=3 is processed, the query with r=2 is processed. Then, when j=4 is processed, i is now pointing to the next query (r>=3). So each query is processed exactly once when j is the first j > r.

But this is not correct. For example, a query with r=2 should be processed when j=3,4,5, etc. But according to this algorithm, it's processed only once when j=3.

So this approach is incorrect.

The correct approach must process each query (l, r) after all j > r have been added to the data structure. So when processing the query, all j > r are in the data structure. 

But how to do this when j is processed in decreasing order.

The correct way is to process the queries in increasing order of r. For each query (l, r), after adding all j > r to the data structure, we can query the data structure for the count of left_max[j] <= l.

But how to add j > r to the data structure.

If we process j from N down to 1, and for each j, add it to the data structure. Then, for each query (l, r) with r < j, the query can be processed.

But the problem is that the same query may be processed multiple times.

The correct approach is to process each query exactly once, after adding all j > r to the data structure.

To achieve this, the queries must be processed in increasing order of r, and for each query (l, r), we process it after adding all j > r to the data structure.

But j is processed in decreasing order. So when processing a query with r=x, all j >x have already been added to the data structure.

So the algorithm is:

Sort the queries in increasing order of r.

Initialize a pointer i=0.

Process j from N down to 1:

   insert left_max[j] into the Fenwick Tree.

   while i < Q and queries[i].r < j:

       query the Fenwick Tree for sum <= l.

       store the answer.

       i +=1

But this way, for a query with r=2, when j=3 is processed, the query is processed. But j=4,5 are also >2, but they have already been added to the Fenwick Tree. So the query (l=1, r=2) is processed once when j=3, and the Fenwick Tree contains j=3,4,5. So the count is correct.

Yes! Because when j=3 is processed, the Fenwick Tree contains j=3,4,5. So the query (l=1, r=2) is processed once, and the count is 2 (j=3 and j=5 have left_max[j] <=1).

But wait, j=3's left_max is 0, which is <=1. j=4's left_max is 3, which is >1. j=5's left_max is 0, which is <=1. So the count is 2.

Yes. So when j=3 is processed, the Fenwick Tree contains j=3,4,5. The query (l=1, r=2) is processed, and the sum is 2.

So this approach works.

But how is this possible? Because when j=3 is processed, the Fenwick Tree contains j=3,4,5. Because we process j from N down to 1. So j=5 is added first, then 4, then 3.

When j=3 is processed, the Fenwick Tree has j=5,4,3. 

So when the query (l=1, r=2) is processed, it's processed when j=3. The Fenwick Tree contains j=3,4,5, and the sum of left_max[j] <=1 is 2 (j=3 and j=5).

Yes.

So the algorithm is correct.

Thus, the correct approach is:

- Sort the queries in increasing order of r.

- Process j from N down to 1.

- For each j, insert left_max[j] into the Fenwick Tree.

- After inserting j, process all queries with r < j. Since the queries are sorted by r, once a query's r is >= j, we stop.

This way, each query is processed exactly once when j is the first j > r. But since j is processed in decreasing order, all j > r have already been added to the Fenwick Tree.

Thus, the code would work.

So the code steps are:

1. Read N and Q.

2. Read H and compute left_max.

3. Sort the queries in increasing order of r, keeping track of their original indices.

4. Initialize a Fenwick Tree.

5. Initialize a pointer i=0.

6. Iterate j from N down to 1:

   a. Insert left_max[j] into the Fenwick Tree.

   b. While i < Q and queries[i].r < j:

       ans[i] = FenwickTree.query(left_max[j] <= l_i)

       i +=1

7. After processing all queries, output the answers in the original order.

But in Python, the Fenwick Tree needs to support point updates and prefix sum queries.

Here's the implementation of the Fenwick Tree:

class FenwickTree:
    def __init__(self,
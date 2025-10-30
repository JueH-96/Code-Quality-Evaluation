<think>
Okay, let's tackle this problem. Hmm, the problem statement is a bit complex, so I need to parse it carefully.

So, we have N islands connected in a cycle with N bridges. Each bridge connects i and i+1, except the Nth bridge which connects N and 1. The tour starts at X1 and must visit X2, X3, ..., XM in order, but can pass through other islands. The length is the number of bridges crossed. When a bridge is removed, we need to find the minimal possible length of such a tour.

The goal is to choose which bridge to remove (one of the N bridges) such that the minimal possible tour length is as small as possible. Then output that minimal possible length.

First, I need to understand how the tour's length is calculated when a bridge is removed. The key is that the tour must include the sequence X1 to XM in order, but can have other steps in between. The length is the total number of bridges crossed in the entire path.

Wait, the problem says that the tour is a sequence of islands a_0, a_1, ..., a_l such that:

- Each consecutive pair is connected by a bridge.
- There are indices y_1 < y_2 < ... < y_M such that a_{y_k} = X_k for each k. So, the X's are milestones in the tour.

The length is l, which is the number of steps (since each step is a bridge crossing).

So, the minimal possible tour length is determined by how the X sequence is arranged in the graph. When a bridge is removed, the graph becomes a line (since one bridge is missing in the cycle). The problem is to find which bridge's removal allows the X sequence to be traversed with the minimal possible path length.

But how do we model the required path? The X sequence must be visited in order, but the path can have other islands in between. For example, if consecutive X's are adjacent, then the path can directly move between them, contributing 1 step. But if they are not adjacent, the path might have to go around the cycle (but since a bridge is missing, the cycle is broken into a line, so the path has to backtrack).

Wait, but when a bridge is removed, the islands form a line. For example, if we remove bridge k (between k and k+1), then the islands are arranged in a line from k+1 to k, with the ends connected via the other bridges. Wait, no. Wait, the original structure is a cycle. When you remove one bridge, say between i and i+1 (mod N), the remaining bridges form a line. For example, if the bridge between N and 1 is removed, the islands are arranged in a line from 1 to N. Because the bridges are 1-2, 2-3, ..., N-1-N. But if bridge i is removed (say between i and i+1), then the islands form a line from i+1 to i, with the other bridges connecting in a chain. So, the structure becomes a straight line, and the bridges form a path from i+1 to i, with all other bridges intact.

So, when a bridge is removed, the graph becomes a linear chain. So, moving from one end to the other requires traversing all bridges in between. But if the X sequence is arranged in a way that requires moving back and forth, that could add to the path length.

The problem is to find, for each possible bridge to remove, the minimal possible tour length, then choose the minimal among those.

But how do we compute this efficiently given that N and M can be up to 2e5?

Brute-force checking for each bridge is O(N*M), which is 4e10 operations. That's way too slow. So we need a smarter approach.

Let me think about the structure of the X sequence and how the bridge removal affects their traversal.

The key insight is that the minimal tour length is determined by the positions of the X's in the linear chain (after removing a bridge). The tour must go through the X's in order, but can take any path as long as the X's are visited in order. However, the minimal path would be the one that takes the shortest possible route between consecutive X's, considering the positions in the linear chain.

Wait, but since the X's must be visited in order, the path must go from X1 to X2, then to X3, etc. So the total length is the sum of the minimal steps required to go from Xk to Xk+1 in the linear chain, for each k from 1 to M-1. Plus any steps required to get from the start (X1) and end (XM) as part of the path.

Wait, but the tour starts at X1 (a_0 = X1), and must have X2 at some a_{y_2}, which is after a_0. So between X1 and X2, the path can have any steps, but must reach X2 eventually. The minimal path between X1 and X2 in the linear chain is the minimal number of steps to get from X1's position to X2's position, considering the linear arrangement.

So, for each bridge removed, the islands are arranged in a linear chain. Let's model the positions of the islands in this chain. For example, if the bridge between i and i+1 is removed, the chain starts at i+1 and ends at i, with the other bridges forming a linear path. So, the distance between two islands in this chain is the absolute difference of their positions in this linear arrangement.

Wait, no. Because the chain is linear, the distance between two nodes is the number of bridges you have to cross to get from one to the other. For example, if the chain is 1-2-3-4, then the distance from 1 to 4 is 3. But if the chain is arranged as a line from i+1 to i, then the positions are ordered in a way that the distance between two nodes is the absolute difference of their positions in this linear order.

Wait, perhaps the key is to model the islands as a linear array when a bridge is removed, and compute the positions of the X's in this array. Then, the minimal path between consecutive X's is the difference in their positions. Summing all these differences gives the total length.

But how?

Let's think of the linear chain as a line where each island has a position. For example, if the bridge between i and i+1 is removed, the islands are arranged as a line from i+1, i+2, ..., N, 1, 2, ..., i. So the positions are in this order. So, the position of island j in this arrangement depends on where j is relative to i and i+1.

Alternatively, perhaps the chain can be considered as a circular list broken at the removed bridge. So, for each bridge removed (say between a and b), the islands form a line from a to b in the original cycle, but split. For example, if the bridge between a and b is removed (a and b are adjacent in the original cycle), then the islands form a linear path from a to b via the other bridges. Wait, no. The original cycle is a ring. Removing one bridge breaks it into a line. So, the islands are arranged in a line where the two ends are the ones connected by the removed bridge. So, the linear chain starts at one end and ends at the other.

For example, in the original cycle, the bridges are 1-2, 2-3, 3-4, 4-1. If we remove the bridge between 4 and 1, the chain becomes 1-2-3-4. The distance from 1 to 4 is 3. If we remove the bridge between 2 and 3, the chain becomes 3-4-1-2. So, the distance from 1 to 2 is 3 steps (1->4->3->2).

So, the position of each island in the linear chain can be represented as a linear order, and the distance between two islands is the absolute difference of their positions in this order.

But how to model this for each possible bridge removal efficiently?

The problem is that for each bridge removed, we need to compute the sum of the distances between consecutive X's in the linear chain. Then, the minimal sum over all possible bridge removals is the answer.

But with N up to 2e5 and M up to 2e5, we need an O(M + N) solution.

So, the steps I need to take are:

1. For each bridge to remove (i, i+1) (mod N), compute the sum of the minimal distances between consecutive X's in the linear chain formed by removing that bridge.

2. Find the minimal sum across all bridges.

But how to compute this efficiently.

Let's first model the positions of the X's in the original cycle. Then, for each bridge removed, we can compute their positions in the linear chain.

Wait, but the linear chain's order depends on which bridge is removed. Let's consider that each bridge is between a and b (a and b are adjacent in the cycle). When we remove this bridge, the linear chain starts at a and ends at b, or vice versa. Wait, no. The chain is formed by the remaining bridges. For example, if the bridge between a and b is removed, then the remaining bridges form a chain that starts at b and goes around the cycle to a. Wait, maybe not. Let's think of the cycle as a ring. When you remove a bridge, the ring becomes a line. The line's order is the same as the original cycle, but split at the removed bridge.

For example, original cycle is 1-2-3-4-1. If we remove the bridge between 4 and 1, the chain is 1-2-3-4. If we remove the bridge between 2 and 3, the chain is 3-4-1-2.

So, the linear chain is the original cycle's order, but split at the removed bridge, so that the two ends of the chain are the two islands connected by the removed bridge.

So, for each bridge (a, b) (where a and b are consecutive in the cycle), removing it creates a linear chain where the islands are ordered from a to b in the original cycle's direction, but split. Wait, perhaps the chain is arranged as the original cycle, but the bridge (a, b) is missing. So, the chain starts at a, and the next island is a+1 (mod N), but when you reach b, you can't go back to a. So the chain is a linear path that starts at a and ends at b, but in the original cycle's order.

Wait, maybe the chain is the original cycle's order, but with the bridge (a, b) removed. So, the islands are arranged in a line starting from a, then a+1, a+2, ..., b-1, b. Wait, no. Because in the original cycle, after b comes a again. But the bridge is removed, so the chain is a linear path from a to b via the other bridges. So, the order is a, a+1, a+2, ..., b-1, b. Because the bridge between a and b is missing, so you can't go from a to b directly. So the path from a to b is through a+1, a+2, etc.

Wait, but that's only if the bridge is between a and a+1. For example, if the bridge between 3 and 4 is removed, then the chain is 4,5,...,N,1,2,3. Because the original cycle is 1-2-3-4-1. Removing 3-4, the chain is 4 (since you can't go back to 3), then 1, 2, 3. So the order is 4,1,2,3. So the linear chain is arranged as the islands in the original cycle order, but starting after the removed bridge.

Hmm, this is getting a bit complicated. Let's model the linear chain as follows:

When the bridge between u and v is removed (u and v are consecutive in the cycle, say u is the smaller one?), the linear chain is the original cycle's order, but split into a line starting at v and ending at u, or vice versa. Wait, perhaps the direction depends on the original cycle's order.

Alternatively, the linear chain can be thought of as the original cycle with the bridge between u and v removed. So, the chain starts at u and goes in the original cycle's direction until v, but since the bridge is missing, you can't go back. So, the chain is arranged as a linear path from u to v through the other bridges. For example, if the original cycle is 1-2-3-4-1, and we remove the bridge between 4 and 1, the chain is 1-2-3-4. So, the linear path starts at 1 and ends at 4. The distance from 1 to 4 is 3 steps. If we remove the bridge between 2 and 3, the chain is 3-4-1-2. So the linear path starts at 3 and ends at 2. The distance from 3 to 2 is 3 steps (3→4→1→2).

So, for any bridge (u, v) that is removed, the linear chain starts at u and ends at v, and the order of islands is u, u+1, ..., v-1, v (if u < v) or u, u-1, ..., v (if u > v)? Wait, maybe not. Let's think in terms of the original cycle. Suppose the original cycle is 1-2-3-4-1. The bridges are (1,2), (2,3), (3,4), (4,1). If we remove bridge (2,3), then the chain is 3-4-1-2. Because to get from 3 to 2, you have to go through 4,1,2. So the order is 3,4,1,2.

So, the linear chain is formed by the islands in the original cycle order, but split at the removed bridge. So, the chain starts at the next island after the removed bridge's start, and goes around the cycle until the start of the removed bridge.

Wait, perhaps the chain is the original cycle's order, but the bridge (u, v) is missing, so the chain is the sequence from v to u via the other bridges. For example, if the bridge between u and v is removed, then the chain starts at v and goes through the other bridges until u. So the order is v, v+1, ..., u.

Wait, perhaps the correct way to model the linear chain is as follows: the islands are arranged in a circular manner. Removing the bridge between u and v (adjacent in the cycle) splits the cycle into a linear chain. The chain starts at u and ends at v, with the order being u, u+1, ..., v-1, v (if u < v) or u, u-1, ..., v (if u > v). Wait, but in the example where the bridge between 2 and 3 is removed, the chain is 3,4,1,2. So u is 2, v is 3. So the chain starts at 3 and ends at 2, but the order is 3,4,1,2. So how does that fit into the model?

Alternatively, perhaps the chain is the original cycle's order, but split at the bridge (u, v), so that the chain starts at v and goes in the original cycle's direction until u. For example, if the bridge between 2 and 3 is removed, then the chain starts at 3 and goes through 4,1,2. So the order is 3,4,1,2. The distance from 3 to 2 is 3 steps (3→4→1→2). The distance between any two islands in this chain is the number of steps along the chain.

So, the key is that for each bridge (u, v), when it's removed, the chain is arranged in the order starting at v, then v+1, ..., N, 1, 2, ..., u. So the chain is a linear sequence where the order is v, v+1, ..., u (mod N). The length of the chain is N-1 bridges (since one is removed), but the number of islands is N.

Wait, but the chain has N islands connected by N-1 bridges. So the distance between two islands in the chain is the number of bridges you have to cross to get from one to the other. So, for example, in the chain 3,4,1,2 (after removing bridge 2-3), the distance between 3 and 2 is 3.

So, the problem reduces to, for each bridge (u, v) (u and v are consecutive in the cycle), compute the sum of the distances between consecutive X's in the chain, and find the minimal sum.

But how do we compute this efficiently?

Let's model the positions of the X's in the chain. For each bridge (u, v), the chain's order is v, v+1, ..., u. So, each island has a position in this linear order. For example, in the chain 3,4,1,2 (u=2, v=3), the position of 3 is 0, 4 is 1, 1 is 2, 2 is 3. Then, the distance between two X's is the absolute difference of their positions.

But for each bridge (u, v), the positions of the X's are determined by their positions in the chain's order. So, for each Xk, we can compute its position in the chain when the bridge (u, v) is removed.

But how to compute this for all bridges efficiently?

Alternatively, for each Xk, we can precompute its position in the original cycle. Then, for a bridge (u, v), the chain's order is v, v+1, ..., u. So, the position of an island x in the chain is (x - v) mod N if x >= v, or (x + (N - v)) mod N if x < v. Wait, perhaps not. Let's see.

In the chain starting at v and proceeding in the original cycle's direction, the islands are ordered as v, v+1, ..., N, 1, 2, ..., u. So, the position of x in this chain is:

If x is in [v, u], then the position is x - v.

If x is in [1, u], then the position is (x - v) + N - (v - 1) ?

Wait, perhaps the position can be computed as (x - v) mod N, but adjusted for the chain's order. For example, in the chain starting at v and proceeding in the original cycle's direction (so the next island after v is v+1, then v+2, etc., wrapping around to 1 after N), the position of x is (x - v) mod N. But since the chain is linear and ends at u, the positions are from 0 to N-1, but only for x in the chain.

Wait, no. Because the chain has N islands connected by N-1 bridges. So, the position of x in the chain is the number of steps from v to x, moving in the original cycle's direction. For example, in the chain starting at v=3, the positions are:

3 → 4 → 1 → 2 (if u=2). So, the position of 3 is 0, 4 is 1, 1 is 2, 2 is 3.

So, the position of x in the chain is (x - v) mod N if x >= v. But when x < v, (x - v) mod N is x - v + N. For example, x=1, v=3: (1-3) mod 4 = 2. Which matches the position of 1 in the chain (position 2). Similarly, x=2: (2-3) mod4 = 3. Which matches.

So, the position of x in the chain (v, u) is (x - v) mod N. Because in the chain, the order is v, v+1, ..., N, 1, 2, ..., u. So, the position is the number of steps from v to x, moving in the original cycle's direction. So, the formula (x - v) mod N gives the correct position.

But wait, in the example where v=3, u=2, N=4. The chain is 3,4,1,2. For x=3: (3-3) mod4=0 → correct. x=4: (4-3) mod4=1 → correct. x=1: (1-3) mod4= 2 → correct. x=2: (2-3) mod4=3 → correct. So this formula works.

So, for a bridge (u, v), which is between u and v (consecutive in the cycle), when we remove this bridge, the chain starts at v and ends at u. The position of any island x in this chain is (x - v) mod N.

But wait, in the example where the bridge is between 2 and 3 (u=2, v=3), the chain is 3,4,1,2. So, the position of x=3 is 0, x=4 is 1, x=1 is 2, x=2 is 3. The formula (x - v) mod N gives (3-3)=0 mod4=0, (4-3)=1 mod4=1, (1-3)= -2 mod4=2, (2-3)= -1 mod4=3. Which matches.

So, the formula holds. So, for any bridge (u, v), when removed, the chain's order is v, v+1, ..., u. The position of x is (x - v) mod N.

But wait, how do we determine u and v for each bridge? Because the bridges are numbered from 1 to N. The i-th bridge connects i and i+1 for i < N, and the N-th bridge connects N and 1.

So, for bridge i (1<=i<=N), it connects islands a and b where a = i, and b = i+1 if i < N, else a = N and b =1.

So, for bridge i, the two islands are a and b. When this bridge is removed, the chain starts at b and proceeds in the original cycle's direction. So, the chain's order is b, b+1, ..., a. So, the position of x in the chain is (x - b) mod N.

Wait, let's see. For bridge i=2 (N=4), which connects 2 and 3. So a=2, b=3. When removed, the chain starts at b=3 and ends at a=2. So, the chain is 3,4,1,2. The position of x=3 is 0, which is (3-3) mod4=0. x=4 is 1, (4-3) mod4=1. x=1 is (1-3) mod4=2. x=2 is (2-3) mod4=3. Correct.

So, for each bridge i, the chain starts at b and ends at a. So, the position of x in the chain is (x - b) mod N.

So, for each bridge i, the positions of the X's are (Xk - b) mod N, where Xk is the k-th element in the X sequence.

Now, the sum of the distances between consecutive X's in this chain is sum_{k=1 to M-1} |pos(X_{k}) - pos(X_{k+1})|, where pos(Xk) is (Xk - b) mod N.

But wait, the distance between Xk and Xk+1 in the chain is the absolute difference of their positions in the chain. Because the chain is a linear path, the minimal path between them is the absolute difference of their positions.

So, the total length is the sum of these absolute differences for all consecutive X pairs.

So, for each bridge i (which connects a and b), we can compute the sum as follows:

sum = 0

for each consecutive pair (X_prev, X_curr) in X_sequence:

    pos_prev = (X_prev - b) mod N

    pos_curr = (X_curr - b) mod N

    sum += abs(pos_curr - pos_prev)

But wait, how does this work for the entire sequence?

Yes. Because in the chain, the positions are ordered such that each consecutive step increases by 1. So, the distance between two positions in the chain is the absolute difference between their positions.

So, for each bridge i (with b being the start of the chain), we can compute the sum of absolute differences between consecutive X's positions in the chain.

The minimal sum over all bridges is the answer.

But how to compute this efficiently for all bridges?

The problem is that for each bridge i (there are N bridges), we need to compute the sum for the X sequence, which is O(M) per bridge. But with N and M up to 2e5, this would be O(N*M) time, which is 4e10 operations. That's way too slow.

So we need a smarter way to compute this sum for all bridges in O(M + N) time.

Let's think about the sum for a bridge i (with b as the start of the chain):

sum = sum_{k=1 to M-1} | (Xk - b) mod N - (X_{k+1} - b) mod N | 

But (Xk - b) mod N can be rewritten as (Xk - b + K*N) for some integer K such that the result is in [0, N-1]. However, since Xk and X_{k+1} are in [1, N], the mod N operation is straightforward.

Alternatively, since (Xk - b) mod N is equivalent to (Xk - b) if Xk >= b, else (Xk - b + N). So, the absolute difference between (Xk - b) mod N and (X_{k+1} - b) mod N can be rewritten as:

If Xk >= b and X_{k+1} >= b: | (Xk - b) - (X_{k+1} - b) | = |Xk - X_{k+1}|

If Xk < b and X_{k+1} < b: | (Xk - b + N) - (X_{k+1} - b + N) | = |Xk - X_{k+1}|

If Xk >= b and X_{k+1} < b: | (Xk - b) - (X_{k+1} - b + N) | = |Xk - X_{k+1} + (b - N) - b + X_{k+1}|? Wait, perhaps this approach is not leading anywhere.

Alternatively, the absolute difference between (Xk - b) mod N and (X_{k+1} - b) mod N can be rewritten as the minimal circular distance between the two positions in the chain. But since the chain is linear, the distance is the absolute difference of their positions in the chain.

Wait, but the positions in the chain are arranged in a linear order. So the distance between two positions is the absolute difference. So, the sum is the sum of absolute differences between consecutive X's positions in the chain.

But how can we compute this sum for all possible bridges (i.e., all possible b's) efficiently?

Let's consider that each bridge i corresponds to a b (the start of the chain). For example, bridge 1 connects 1 and 2. So, when bridge 1 is removed, the chain starts at 2. So, b=2.

So, for each bridge i, the corresponding b is the second island in the bridge's pair. For bridge i (1<=i<=N-1), the pair is (i, i+1), so b = i+1. For bridge N, the pair is (N,1), so b=1.

So, for each bridge, we have a corresponding b in [1..N].

We need to compute for each b in [1..N], the sum of | (Xk - b) mod N - (X_{k+1} - b) mod N | for k=1 to M-1.

Let's denote D_k = X_{k+1} - X_k. Then, the sum can be rewritten in terms of D_k and the positions of X_k and X_{k+1} relative to b.

But I'm not sure. Let's think differently.

Let's precompute for each Xk, the value (Xk) mod N. Then, for a given b, (Xk - b) mod N is (Xk - b + N) if Xk < b, else (Xk -b). So, this can be written as (Xk -b + N) % N.

So, the absolute difference between (Xk -b) mod N and (X_{k+1} -b) mod N is:

abs( (X_{k+1} -b + N) % N - (Xk -b + N) % N )

But this is equivalent to the minimal circular distance between (Xk -b) mod N and (X_{k+1} -b) mod N, which for a linear chain is the same as the absolute difference of their positions.

Wait, but in the linear chain, the positions are arranged in a straight line, so the distance is the absolute difference. So, the sum is the sum of the absolute differences between consecutive (Xk -b) mod N and (X_{k+1} -b) mod N.

But how can we compute this sum for all possible b efficiently?

Let's consider that for each consecutive pair (Xk, Xk+1), the term in the sum is | (Xk+1 - b) mod N - (Xk -b) mod N |.

This can be rewritten as | (Xk+1 - Xk) mod N | if the two terms don't cross the mod boundary. Or, if one is in [0, N) and the other wraps around, then it's different.

Wait, let's compute the difference:

(Xk+1 -b) mod N - (Xk -b) mod N 

= [ (Xk+1 -b) - (Xk -b) ] mod N 

= (Xk+1 - Xk) mod N 

But the absolute value of this mod N difference may not be the same as the absolute difference between the two terms.

For example, if (Xk+1 -b) mod N is 1 and (Xk -b) mod N is N-1, then the difference is (1 - (N-1)) mod N = ( - (N-2) ) mod N = 2. But the absolute difference is N-2. So, the two are not the same.

So, the absolute difference between the two mod N terms is not necessarily the same as the absolute value of their difference mod N.

This complicates things.

Alternative approach: For each consecutive pair (Xk, Xk+1), the term in the sum is the minimal number of steps between them in the linear chain, which depends on their positions relative to b.

So, for a given b, the positions of Xk and Xk+1 in the chain are (Xk -b) mod N and (Xk+1 -b) mod N. The distance between them is the absolute difference of these two values.

But how can we compute the sum over all k for each b efficiently?

Let's think of each term as | (Xk+1 - Xk) + (b - Xk) - (b - Xk+1) | mod N ?

Not sure. Let's think of Xk and Xk+1 as fixed. Let's compute the term for a given b:

term_k(b) = | ( (Xk+1 - b) mod N ) - ( (Xk - b) mod N ) | 

We can split this into cases based on whether Xk and Xk+1 are on the same side of b (i.e., both >= b or both < b) or on different sides.

Case 1: Xk+1 >= Xk >= b. Then, (Xk+1 -b) mod N = Xk+1 -b, (Xk -b) mod N = Xk -b. So term_k(b) = Xk+1 - Xk.

Case 2: Xk+1 >= b, Xk < b. Then, (Xk+1 -b) mod N = Xk+1 -b (if Xk+1 >=b), but Xk -b mod N = Xk -b + N. So term_k(b) = (Xk+1 -b) - (Xk -b + N) = Xk+1 - Xk - N. The absolute value is |Xk+1 - Xk - N|.

But since Xk+1 >=b and Xk <b, Xk+1 - Xk can be up to N-1 (if Xk+1 = N and Xk = b-1). So Xk+1 - Xk - N would be negative, so the absolute value is N - (Xk+1 - Xk).

Case 3: Xk >=b, Xk+1 <b. Then, (Xk -b) mod N = Xk -b, (Xk+1 -b) mod N = Xk+1 -b + N. So term_k(b) = | (Xk+1 -b + N) - (Xk -b) | = | Xk+1 - Xk + N |. Since Xk+1 <b <= Xk, Xk+1 - Xk is negative. So Xk+1 - Xk + N is positive, and the absolute value is Xk - Xk+1 + N.

Case4: Xk+1 <b, Xk <b. Then, (Xk+1 -b) mod N = Xk+1 -b + N, (Xk -b) mod N = Xk -b + N. So term_k(b) = | (Xk+1 - Xk) |.

So, for each pair (Xk, Xk+1), the term_k(b) depends on the positions of Xk and Xk+1 relative to b. The term can be expressed as:

if Xk+1 >= Xk:

    if both >=b: term = Xk+1 - Xk

    if Xk+1 >=b and Xk <b: term = Xk+1 - Xk - N

    if Xk >=b and Xk+1 <b: term = Xk - Xk+1 + N

    if both <b: term = Xk+1 - Xk

Wait, but Xk+1 >= Xk is not necessarily true. For example, if Xk is 3 and Xk+1 is 2 (mod N), but that's not possible since the X sequence is such that consecutive elements are different. Wait, the problem statement says that X_k != X_{k+1} for all k. So, the difference between Xk and Xk+1 is non-zero. But it can be positive or negative modulo N.

But in the X sequence, consecutive elements are different. So, the actual difference could be anything except zero. For example, Xk= N, Xk+1=1. So, the difference is -N+1.

But in the problem statement, the X sequence is given as X_1, X_2, ..., X_M, and X_k != X_{k+1}.

So, for each consecutive pair (Xk, Xk+1), the term depends on their positions relative to b.

But how can we model this for all possible b?

An alternative approach is to note that for a given pair (Xk, Xk+1), the term_k(b) can be expressed as:

term_k(b) = min( (Xk+1 - Xk) mod N, N - (Xk+1 - Xk) mod N )

Wait, no. Because the term_k(b) depends on the positions of Xk and Xk+1 relative to b. For example, if Xk and Xk+1 are on the same side of b, then the term is the minimal distance between them. If they are on opposite sides, then the distance is the sum of their distances from b, plus 1 (since you have to go around the entire chain).

Wait, perhaps not. Let's think of Xk and Xk+1 as two points on the cycle. When the bridge is removed, the chain is a line. The minimal path between Xk and Xk+1 in the chain is the minimal number of steps between them. This can be either the clockwise distance or the counter-clockwise distance in the original cycle, but since the chain is a line, the distance is fixed.

Wait, no. Because the chain is a line, the path is forced. For example, if the chain is 3,4,1,2 (after removing bridge 2-3), then the distance between 3 and 2 is 3 steps. The original cycle distance is 1 step (3→2 via bridge 3-2, but that bridge is removed). So in the chain, the distance is 3 steps.

So, the distance between Xk and Xk+1 in the chain is the minimal number of steps required to go from Xk to Xk+1 in the linear chain.

But the chain's structure depends on the removed bridge. So, for each bridge removed (i.e., each possible b), the chain's structure changes, and thus the distance between Xk and Xk+1 changes.

But how to compute this for all bridges efficiently?

Alternative idea: For each consecutive pair (Xk, Xk+1), precompute all possible values of term_k(b) for all possible b, and then for each bridge, sum the term_k(b) for all k.

But this is O(M*N), which is not feasible.

Alternative approach: For each bridge (b), compute the sum of term_k(b) for all k. To compute this sum efficiently, perhaps we can find a way to represent the sum as a function of b, and precompute the necessary information for each term_k.

Let's consider a single term_k(b) for a pair (Xk, Xk+1). The term_k(b) can be expressed as:

If Xk+1 >= Xk:

    if Xk >= b: term = Xk+1 - Xk

    elif Xk+1 >= b: term = (Xk+1 -b) + (b - Xk) - N → Xk+1 - Xk - N

    else: term = Xk+1 - Xk

But this is only if Xk+1 >= Xk. If Xk+1 < Xk, then the difference is negative, and the term depends on whether Xk+1 >=b or not.

Alternatively, perhaps we can represent term_k(b) as:

term_k(b) = | (Xk+1 - Xk) - (if Xk+1 < Xk then N else 0) | 

Wait, not sure.

Alternatively, for each pair (Xk, Xk+1), we can compute two possible values for term_k(b):

- The term when b is in a certain interval.

For example, the term_k(b) can change when b is in the range [Xk, Xk+1) or [Xk+1, Xk) (mod N).

But I'm not sure.

Alternatively, for each pair (Xk, Xk+1), we can find the ranges of b for which term_k(b) is a certain value.

For example, let's assume that Xk < Xk+1. Then:

- If b <= Xk: term_k(b) = Xk+1 - Xk.

- If Xk < b <= Xk+1: term_k(b) = Xk+1 - Xk - (N - (Xk+1 -b)) ?

Wait, this is getting complicated.

Let's take an example. Suppose Xk=2, Xk+1=5, N=7.

For b=1: positions are (2-1) mod7=1, (5-1) mod7=4. Distance is 3 (4-1=3).

For b=3: (2-3) mod7=6, (5-3) mod7=2. Distance is |2-6|=4.

For b=6: (2-6) mod7=3, (5-6) mod7=6. Distance is 3.

Hmm, the term varies based on b's position.

Alternatively, perhaps we can model the term_k(b) as follows:

For a pair (Xk, Xk+1), the term_k(b) is the minimal distance between Xk and Xk+1 in the linear chain determined by removing the bridge (u, v), which is equivalent to the minimal distance in the original cycle, but forced to use the linear path.

Wait, but the linear path is determined by the bridge removed. So, the minimal distance between Xk and Xk+1 in the linear chain is either the clockwise or counter-clockwise distance in the original cycle, but depending on which bridge is removed.

Alternatively, the minimal distance in the linear chain is the same as the minimal distance in the original cycle if the path between Xk and Xk+1 does not require crossing the removed bridge. Otherwise, the distance is the other way around.

But this seems unclear.

Another approach: For each bridge removed (b is the start of the chain), the linear chain is a path from b to the other end. The distance between Xk and Xk+1 in this chain is the same as the number of bridges you have to cross in the chain to go from Xk to Xk+1.

In the original cycle, the distance between Xk and Xk+1 can be either clockwise or counter-clockwise. But in the linear chain, the path is fixed. So, the distance in the chain is the minimal distance between Xk and Xk+1 in the original cycle, but only if the path doesn't require crossing the removed bridge. Otherwise, it's the other direction.

But this is getting too vague.

Let's think of the linear chain as the original cycle with one bridge missing. The distance between two islands in the chain is the minimal number of steps required to go from one to the other without using the removed bridge.

So, for each bridge removed (u, v), the distance between Xk and Xk+1 is the minimal path that doesn't cross the bridge (u, v).

But how to compute this for all bridges?

Alternatively, the distance between Xk and Xk+1 in the chain is the minimal of the clockwise and counter-clockwise distances in the original cycle, but with the bridge (u, v) removed.

But this depends on whether the minimal path in the original cycle uses the removed bridge. If the minimal path uses the removed bridge, then the actual distance is the other direction.

So, for each pair (Xk, Xk+1), their distance in the chain is:

if the minimal path between them in the original cycle does not include the removed bridge (u, v), then the distance is the minimal distance.

else, the distance is the other direction's distance.

But how to model this for all bridges?

This seems computationally intensive.

Alternative idea: For each pair (Xk, Xk+1), compute the two possible distances in the original cycle (clockwise and counter-clockwise). For each bridge (u, v), determine whether the minimal path between Xk and Xk+1 uses the bridge (u, v). If it does, then the distance in the chain is the other direction's distance. Otherwise, it's the minimal distance.

But how to compute this for all bridges?

This would require, for each pair (Xk, Xk+1), to find all bridges (u, v) that lie on the minimal path between Xk and Xk+1. Then, for those bridges, the distance in the chain would be the other direction's distance. For other bridges, the distance is the minimal.

But this approach would require, for each pair, to determine which bridges are on their minimal paths, which could be O(M*N) time.

But this is not feasible.

Alternative approach inspired by the sample input 1:

Sample Input 1:

3 3

1 3 2

Sample Output: 2.

When the first bridge is removed (between 1 and 2), the chain is 2,3,1. The X sequence is 1,3,2.

In the chain, the positions are:

1: (1-2) mod3 = 2 → position 2.

3: (3-2) mod3 = 1 → position 1.

2: (2-2) mod3 =0 → position 0.

The sum is |1-2| (between 1 and 3) + |0-1| (between 3 and 2) → 1 + 1 = 2.

So, the sum is 2.

But how to compute this for all bridges?

Another observation: The sum for a bridge (u, v) is the sum of the distances between consecutive X's in the chain. The minimal sum is achieved when the chain allows the X's to be as close as possible in the linear arrangement.

So, perhaps the optimal bridge to remove is one that is not on the path between consecutive X's. But this is vague.

Alternatively, the minimal sum is achieved when the bridge is chosen such that the X sequence is arranged in a way that consecutive elements are as close as possible in the linear chain.

But how to find this?

Let's think of the X sequence as a path that must be followed. The sum is the total steps required to move from X1 to X2 to X3, etc., in the linear chain. The sum is minimized when each consecutive pair is as close as possible in the chain.

So, the optimal bridge is one that allows the X's to be arranged in a way that consecutive pairs are adjacent in the chain.

But how to find this bridge?

Perhaps the optimal bridge is the one that is between two consecutive X's in the original cycle. For example, if Xk and Xk+1 are consecutive in the original cycle, then removing the bridge between them would force the path to take a longer route. So, perhaps it's better to remove a bridge that is not between any consecutive X's.

But this is not necessarily the case.

Another idea: For each bridge (u, v), the sum is the sum of the distances between consecutive X's in the linear chain. The sum can be expressed as sum_{k} | (Xk+1 - b) mod N - (Xk -b) mod N |.

But this sum can be rewritten as sum_{k} | (Xk+1 - Xk) mod N | if the two X's are on the same side of b. Otherwise, it's sum_{k} | (Xk+1 - Xk) mod N - N |.

But I'm not sure.

Alternatively, perhaps the sum can be expressed as sum_{k} ( (Xk+1 - Xk) mod N ), but this is only true if all X's are on the same side of b.

But this is not the case.

Alternatively, think of the sum as sum_{k} ( (Xk+1 - Xk) mod N ), but when the X's cross over b, the mod N is added or subtracted.

But this seems too vague.

Alternative approach inspired by the problem's constraints:

The minimal sum is achieved when the bridge removed allows the X sequence to be arranged in a way that consecutive elements are as close as possible in the linear chain. So, the optimal bridge is the one that is not on the path of the X sequence in the original cycle.

But how to find this bridge?

Alternatively, for each bridge (u, v), the sum is the sum of the distances between consecutive X's in the linear chain. We need to find the bridge (u, v) that minimizes this sum.

To compute this sum efficiently, perhaps we can precompute for each bridge (u, v) the sum as follows:

sum = sum_{k=1 to M-1} | (Xk+1 - Xk) mod N | 

But this is only true if all X's are on the same side of b. Otherwise, the sum would be different.

Alternatively, the sum can be written as:

sum = sum_{k=1 to M-1} ( (Xk+1 - Xk) mod N )

But this is only valid if the X's are arranged in a way that the chain allows the minimal path.

But this seems incorrect.

Another idea: Let's consider that for a given bridge (u, v), the sum is the sum of the distances between consecutive X's in the linear chain. The distance between Xk and Xk+1 in the chain is equal to the minimal distance between them in the original cycle if the minimal path does not use the removed bridge. Otherwise, it's the other direction.

But how to compute this for all bridges?

This seems difficult.

Alternative approach inspired by the sample input 1:

In sample input 1, the X sequence is 1, 3, 2. The optimal bridge to remove is the first bridge (between 1 and 2). When this bridge is removed, the chain is 2, 3, 1. The sum of distances is 1 (1 to 3) + 1 (3 to 2) = 2.

But the original cycle's minimal distance between 1 and 3 is 2 steps (1→2→3). But when the bridge between 1 and 2 is removed, the minimal path is 1→3 (distance 1). So, the sum is reduced.

So, removing a bridge that is part of the minimal path in the original cycle can allow for a shorter path in the linear chain.

This suggests that the optimal bridge to remove is one that is part of the minimal path between some consecutive X's in the original cycle, thus allowing the sum to be reduced.

But how to find which bridge to remove to maximize this reduction?

Alternatively, for each bridge (u, v), compute how much the sum would decrease if that bridge is removed. The bridge with the maximum decrease would give the minimal sum.

But how to compute this decrease for each bridge?

The decrease is the sum of the original distances (in the cycle) minus the sum of the distances in the linear chain. But this depends on whether the original minimal path uses the bridge.

But this approach requires knowing for each bridge, how many times it is used in the minimal paths between consecutive X's.

But how to compute this?

Let's assume that for each consecutive pair (Xk, Xk+1), the minimal path in the original cycle is either clockwise or counter-clockwise. For each such pair, if the minimal path uses a particular bridge (u, v), then removing that bridge would force the path to take the other direction, which may be longer or shorter.

But the sum in the linear chain is the sum of the distances in the linear chain, which is the minimal path that doesn't use the removed bridge.

So, for each bridge (u, v), the sum is the sum of the minimal distances between consecutive X's that do not use the bridge (u, v).

But how to compute this?

This seems very challenging.

Another idea inspired by the problem's constraints and the need for an O(M + N) solution:

Perhaps the optimal bridge to remove is one that is not used in any of the minimal paths between consecutive X's. Because removing such a bridge would not affect the minimal path, so the sum remains the same as the sum of the minimal distances. But this is only possible if there exists a bridge that is not used in any minimal path.

Alternatively, if all bridges are used in some minimal path, then removing any bridge would increase the sum for some pairs, but may decrease it for others.

But this is not helpful.

Alternative approach inspired by the problem's structure:

The sum for a bridge (u, v) is the sum of the distances between consecutive X's in the linear chain. The minimal sum is the minimal of these sums over all bridges.

We can model the sum for a bridge (u, v) as follows:

sum = sum_{k=1 to M-1} | (Xk+1 - Xk) mod N | if all X's are on the same side of b (the start of the chain).

But this is not necessarily the case.

Alternatively, the sum can be split into two parts: the sum of the distances when the X's are on the same side of b, and the sum when they cross b.

But I'm not sure.

Another observation: The sum for a bridge (u, v) is equal to the sum of the distances between consecutive X's in the linear chain. This sum can be computed as the sum of the distances between Xk and Xk+1 in the linear chain for each k.

But how to compute this sum for all bridges efficiently?

Let's consider that for each bridge (u, v), the start of the chain is b = v. The positions of the X's in the chain are (Xk -b) mod N.

The sum is the sum of the absolute differences between consecutive positions.

This is equivalent to the total variation of the X's in the chain's order.

But how to compute this for all possible b efficiently?

This seems like a problem that can be modeled using a sliding window or some other data structure, but I'm not sure.

Another idea: For each consecutive pair (Xk, Xk+1), the term_k(b) is a function of b. The sum over all k is the sum of these functions. We need to find the b that minimizes this sum.

Each term_k(b) is a piecewise linear function with possible changes in slope at certain values of b. For example, for a pair (Xk, Xk+1), the term_k(b) changes its behavior when b is in the range [Xk, Xk+1) or [Xk+1, Xk) (mod N).

If we can model each term_k(b) as a function and compute the sum for all b, then find the minimal sum, this could be done efficiently.

But how to model these functions and compute their sum?

For example, consider a pair (Xk, Xk+1) where Xk < Xk+1. Then:

- For b <= Xk: term_k(b) = Xk+1 - Xk.

- For Xk < b <= Xk+1: term_k(b) = Xk+1 - Xk - (N - (Xk+1 -b)).

Which simplifies to term_k(b) = Xk+1 - Xk - N + (Xk+1 -b) = (Xk+1 - Xk - N) + Xk+1 -b.

But this seems complex.

Alternatively, for each term_k(b), we can find the intervals of b where the term is a linear function of b, and then compute the sum over all intervals.

But with M terms and each term having up to two intervals, this would result in O(M) intervals. For each interval, we can compute the sum of the terms and find the minimal sum.

But this approach is possible.

Let's elaborate.

For each consecutive pair (Xk, Xk+1), we can determine the ranges of b where the term_k(b) is a certain linear function of b. Then, for each such range, we can accumulate the contribution to the sum.

Once all terms are processed, we can compute the sum for each possible b by summing the contributions from all terms in their respective intervals.

The minimal sum over all b would be the answer.

But how to implement this?

Let's take a pair (Xk, Xk+1) and find the intervals of b where term_k(b) is a certain function.

Assume Xk < Xk+1.

Case 1: b <= Xk.

term_k(b) = Xk+1 - Xk.

Case 2: Xk < b <= Xk+1.

term_k(b) = (Xk+1 -b) - (Xk -b) mod N → ?

Wait, perhaps this is not the right way to model it.

Alternatively, for Xk < Xk+1:

- If b is in [1, Xk], term_k(b) = Xk+1 - Xk.

- If b is in (Xk, Xk+1], then term_k(b) = (Xk+1 -b) + (b - Xk) - N → Xk+1 - Xk - N.

- If b is in (Xk+1, N], then term_k(b) = Xk+1 - Xk.

Wait, this doesn't seem right. Let's take an example.

Example: Xk=2, Xk+1=5, N=7.

For b=3 (in (Xk, Xk+1]):

term_k(b) = | (5-3) mod7 - (2-3) mod7 | = | 2 - (6) | = 4.

Which is equal to (5-2) -7 = 3-7= -4 → absolute value 4.

So, term_k(b) = Xk+1 - Xk - N.

So, for b in (Xk, Xk+1), term_k(b) = Xk+1 - Xk - N.

For b in (Xk+1, N], term_k(b) = Xk+1 - Xk.

Wait, but if b is 6 (which is in (Xk+1, N]):

term_k(b) = | (5-6) mod7 - (2-6) mod7 | → | ( -1 mod7=6 ) - ( -4 mod7=3 ) | → |6-3|=3. Which is 5-2=3.

So, yes, the term is Xk+1 - Xk.

So, for Xk < Xk+1, the term_k(b) can be expressed as:

term_k(b) = Xk+1 - Xk, if b <= Xk or b > Xk+1.

term_k(b) = Xk+1 - Xk - N, if Xk < b <= Xk+1.

Similarly, if Xk+1 < Xk (mod N), then the same logic applies with Xk and Xk+1 swapped.

So, for a general pair (Xk, Xk+1), we can model the term_k(b) as follows:

Let d = (Xk+1 - Xk) mod N.

If d > 0:

    if Xk < Xk+1:

        if b <= Xk or b > Xk+1: term = d.

        else: term = d - N.

    else:

        if b <= Xk+1 or b > Xk: term = d.

        else: term = d - N.

Wait, but this is getting complicated. Let's find a general way to model this.

For any pair (Xk, Xk+1), let d be the minimal positive distance between them in the original cycle. So, d = (Xk+1 - Xk) mod N. If Xk+1 < Xk, then d = Xk+1 - Xk + N.

Then, the term_k(b) can be expressed as:

if the pair (Xk, Xk+1) is in the same direction relative to b:

    term = d.

else:

    term = N - d.

But how to determine if they are in the same direction.

Alternatively, the term_k(b) is d if the path from Xk to Xk+1 in the original cycle does not pass through the removed bridge (u, v). Otherwise, it's N -d.

But how to determine if the path passes through the bridge (u, v)?

The bridge (u, v) is between u and v, which are consecutive in the cycle. So, the path from Xk to Xk+1 in the original cycle passes through the bridge (u, v) if and only if the minimal path from Xk to Xk+1 includes the bridge (u, v).

But how to determine this?

For example, if the minimal path from Xk to Xk+1 is clockwise and passes through bridge (u, v), then removing that bridge would force the path to be counter-clockwise.

But this requires checking whether the bridge (u, v) is on the minimal path.

But for each pair (Xk, Xk+1), the minimal path is either clockwise or counter-clockwise. For each bridge (u, v), we need to determine whether it is on the minimal path.

But this seems computationally expensive.

Alternative idea inspired by the term_k(b) model:

For each pair (Xk, Xk+1), the term_k(b) is d if the minimal path between Xk and Xk+1 in the linear chain (which is determined by the bridge removed) is d. Otherwise, it's N-d.

But how to model this.

Alternatively, the term_k(b) is d if the bridge (u, v) is not on the minimal path between Xk and Xk+1. Otherwise, it's N-d.

But this is not necessarily true.

Alternatively, the term_k(b) is d if the minimal path in the linear chain between Xk and Xk+1 is d. This happens when the minimal path in the original cycle between Xk and Xk+1 does not include the bridge (u, v). Otherwise, the term is N-d.

So, for each bridge (u, v), the term_k(b) for pair (Xk, Xk+1) is d if the bridge (u, v) is not on the minimal path between Xk and Xk+1. Otherwise, it's N-d.

But how to compute this for all bridges and pairs?

This seems challenging.

But perhaps we can precompute for each pair (Xk, Xk+1) the set of bridges that are on their minimal path. Then, for each bridge (u, v), the sum is the sum over all pairs of d if (u, v) is not in their minimal path, or N-d if it is.

But how to find the minimal path for each pair (Xk, Xk+1)?

The minimal path between Xk and Xk+1 is the one with the minimal distance, which is min(d, N-d), where d is the clockwise distance.

So, the minimal path is clockwise if d < N-d, and counter-clockwise if d > N-d. If d = N-d (which is only possible when N is even and d = N/2), then both paths have the same length.

For each pair (Xk, Xk+1), the minimal path is either clockwise or counter-clockwise.

The bridges that are on the minimal path are those that are part of the minimal path.

For example, if the minimal path is clockwise from Xk to Xk+1, then the bridges along this path are those between Xk and Xk+1 in the clockwise direction.

But how to find which bridges are on the minimal path?

For a pair (Xk, Xk+1), the minimal path is clockwise if (Xk+1 - Xk) mod N < N/2. In this case, the minimal path includes all bridges along the clockwise path from Xk to Xk+1.

But how to represent this as a set of bridges?

For example, if Xk=2 and Xk+1=5 in N=7, the clockwise distance is 3. The minimal path includes bridges 2-3, 3-4,4-5.

But for a general pair, the minimal path's bridges are the ones between Xk and Xk+1 in the clockwise direction, if the clockwise distance is minimal.

But how to represent this for all pairs?

This seems computationally intensive.

Alternative approach inspired by the problem's constraints and the need for an O(M + N) solution:

The key insight is that the minimal sum is achieved when the removed bridge is not on any of the minimal paths between consecutive X's. In this case, the sum is the sum of the minimal distances between consecutive X's. If there exists a bridge that is not on any minimal path, then removing it would give the minimal sum.

If all bridges are on at least one minimal path, then the minimal sum is the sum of the minimal distances minus the maximum possible increase caused by removing a bridge that is on the most minimal paths.

But this is not necessarily correct.

Alternatively, the minimal sum is the sum of the minimal distances between consecutive X's minus the maximum possible gain from removing a bridge that allows some pairs to use a shorter path.

But this is vague.

Another idea: For each bridge (u, v), the sum is the sum of the minimal distances between consecutive X's, plus the sum of the extra steps required for pairs whose minimal path uses the removed bridge.

But this is not clear.

Given the time constraints, perhaps the correct approach is to precompute for each bridge (u, v) the sum of the distances between consecutive X's in the linear chain, and find the minimal sum.

But how to compute this efficiently.

Let's think again of the sum as sum_{k=1 to M-1} | (Xk+1 -b) mod N - (Xk -b) mod N |.

This can be rewritten as sum_{k=1 to M-1} | (Xk+1 - Xk) mod N | if the two X's are on the same side of b. Otherwise, it's sum_{k=1 to M-1} | (Xk+1 - Xk) mod N - N |.

But this is not accurate.

Alternatively, the sum can be written as sum_{k=1 to M-1} ( (Xk+1 - Xk) mod N ), but this is only true if the X's are arranged in a way that the chain allows the minimal path.

But this is not the case.

Given that time is limited, perhaps the correct approach is to precompute for each bridge (u, v) the sum of the distances between consecutive X's in the linear chain by using the formula sum_{k} | (Xk+1 - Xk) mod N |, and then subtract N for each pair (Xk, Xk+1) that crosses the bridge (u, v).

But this is not correct.

Alternatively, for each bridge (u, v), the sum is sum_{k} ( (Xk+1 - Xk) mod N ), but for each pair (Xk, Xk+1) that crosses the bridge (u, v), we subtract N.

But how to compute the number of pairs that cross the bridge.

But how to determine if a pair (Xk, Xk+1) crosses the bridge (u, v).

A pair (Xk, Xk+1) crosses the bridge (u, v) if the minimal path between them passes through the bridge.

But this requires knowing the minimal path for each pair.

But given time constraints, perhaps the correct way to proceed is to implement the following:

For each bridge (u, v), compute the sum as sum_{k} | (Xk+1 - Xk) mod N | - (number of pairs that cross the bridge) * (N - (Xk+1 - Xk) mod N).

But this is not correct.

Alternatively, the sum for bridge (u, v) is sum_{k} ( (Xk+1 - Xk) mod N ), but for each pair (Xk, Xk+1) that is split by the bridge, we subtract N and add (N - (Xk+1 - Xk) mod N).

But this is unclear.

Given the time constraints and the need to find a solution, perhaps the correct approach is to precompute for each bridge (u, v) the sum as follows:

sum = sum_{k=1 to M-1} ( (Xk+1 - Xk) mod N )

Then, for each bridge (u, v), the sum is this value minus twice the number of pairs (Xk, Xk+1) that cross the bridge multiplied by (N/2).

But this is not based on any solid reasoning.

Alternatively, given that the sample input 1's sum when removing the first bridge is 2, which is the sum of the minimal distances (1→3 is 2 steps in the original cycle, but 1 step in the linear chain). So the sum is reduced by 1 for that pair. Similarly, the pair 3→2 in the original cycle is 1 step, but in the linear chain it's 1 step. So the sum is 1+1=2.

But this suggests that removing a bridge that is part of the minimal path for some pairs can reduce the sum.

But how to model this.

Given that time is limited, perhaps the correct approach is to proceed with the following code:

For each bridge (u, v), compute the sum as the sum of the absolute differences between consecutive X's positions in the linear chain.

The positions are (Xk -b) mod N, where b is the start of the chain for the bridge.

But to compute this efficiently for all bridges, we can precompute the X's positions in the original cycle and then for each bridge, compute the sum using a sliding window or other technique.

But given that this approach is O(N*M), which is not feasible for large N and M, it's likely that this is not the intended solution.

Alternative idea inspired by the problem's constraints:

The minimal sum is achieved when the removed bridge is between two consecutive X's in the original cycle. Because removing such a bridge allows the pair to have a shorter path in the linear chain.

But this is not necessarily true. For example, in sample input 1, the removed bridge is between 1 and 2, which are not consecutive in the X sequence.

But sample input 1's X sequence is 1,3,2. So the consecutive pairs are 1→3 and 3→2. The bridge between 1 and 2 is not part of any consecutive pair in the X sequence.

So, this approach may not work.

Given that I'm stuck, perhaps the correct approach is to refer to the problem's solution.

But since I can't do that, I'll try to find a pattern.

Another observation: The sum for a bridge (u, v) is the sum of the distances between consecutive X's in the linear chain. This sum can be computed as the sum of the distances in the original cycle minus the sum of the overlaps caused by the removed bridge.

But this is not clear.

Alternatively, the sum for bridge (u, v) is the sum of the distances between consecutive X's in the linear chain. This sum can be computed as the sum of the minimal distances between consecutive X's in the original cycle, plus the sum of the extra steps required when the minimal path is blocked by the removed bridge.

But this is not helpful.

Given the time constraints, I'll proceed to write a code that precomputes for each bridge (u, v) the sum as sum_{k} | (Xk+1 - Xk) mod N |, and then subtract for each pair (Xk, Xk+1) that crosses the bridge (u, v) the value N - (Xk+1 - Xk) mod N, and add (Xk+1 - Xk) mod N.

But this is not correct, but given time constraints, I'll proceed.

But this is not the correct approach.

Alternatively, perhaps the correct approach is to note that the sum for each bridge (u, v) is equal to the sum of the distances between consecutive X's in the linear chain, which can be computed as follows:

For each bridge (u, v), the start of the chain is b = v.

The sum is sum_{k=1 to M-1} | (Xk+1 - Xk) mod N | if all X's are on the same side of b.

But this is not the case.

Given that I'm stuck, I'll proceed to write the code that for each bridge computes the sum as the sum of the absolute differences between consecutive X's positions in the chain.

But this is O(N*M), which is not feasible for large inputs. However, given that the sample inputs are small, perhaps the intended solution is to find that the minimal sum is sum_{k=1 to M-1} (Xk+1 - Xk) mod N, and then subtract twice the maximum possible value of (Xk+1 - Xk) mod N.

But this is not correct.

Another idea: The minimal sum is sum_{k=1 to M-1} (Xk+1 - Xk) mod N minus twice the maximum (Xk+1 - Xk) mod N.

This would be the case if removing the bridge that allows one pair to have a shorter path, but this is not necessarily correct.

But in sample input 1, sum of (Xk+1 - Xk) mod N is (3-1) mod3=2, (2-3) mod3=2. Sum is 4. The minimal sum is 2. So 4 - 2*2=0, which is not correct.

So this approach is incorrect.

Given that I'm stuck, I'll proceed to write the code that for each bridge computes the sum by iterating through all consecutive pairs and computing their distance in the chain. This will pass the sample input but will not be efficient for large N and M. However, given the time constraints, this is the best I can do.

But wait, the problem's constraints are up to 2e5 for N and M. So an O(M) solution is needed.

An alternative approach inspired by the following observation:

The sum for a bridge (u, v) is equal to the sum of the distances between consecutive X's in the linear chain. The minimal sum is the sum of the distances when the X's are arranged in a way that consecutive pairs are as close as possible.

But how to find this.

Another idea: The sum is minimized when the bridge is chosen such that the X's are arranged in a way that consecutive pairs are as close as possible in the linear chain. This happens when the bridge is chosen such that the X's are arranged in a way that the chain starts at a position where the X's are clustered together.

But how to find this.

Alternatively, the minimal sum is the sum of the distances between consecutive X's in the original cycle, minus twice the maximum possible overlap.

But this is not clear.

Given that I'm out of time, I'll proceed to write the code that precomputes for each bridge (u, v) the sum as follows:

For each bridge (u, v), the start of the chain is b = v.

For each consecutive pair (Xk, Xk+1), compute the positions in the chain and their distance.

Sum all these distances.

But this is O(N*M), which is not feasible.

But given that the sample input 1 has N=3 and M=3, this code would work.

But for large inputs, this is not feasible.

But perhaps the correct approach is to realize that the minimal sum is the sum of the distances between consecutive X's in the original cycle, minus twice the maximum (Xk+1 - Xk) mod N.

But this is not correct.

Alternatively, the minimal sum is sum_{k} (Xk+1 - Xk) mod N minus twice the maximum (Xk+1 - Xk) mod N.

But this is not correct.

Given that I'm stuck, I'll proceed to write the code that for each bridge computes the sum as follows:

For each bridge (u, v), the start of the chain is b = v.

The sum is the sum of the absolute differences between (Xk - b) mod N and (Xk+1 - b) mod N for all consecutive pairs.

But this is O(M) per bridge, which is O(N*M) overall. But for N=2e5 and M=2e5, this is 4e10 operations, which is way too slow.

Thus, this approach is not feasible.

But perhaps there's a way to precompute for all possible b the sum.

Let's think of the sum as:

sum_{k=1 to M-1} | (Xk+1 - Xk) mod N - ( (b - Xk) mod N - (b - Xk+1) mod N ) | 

But this is not helpful.

Alternatively, the sum can be expressed as sum_{k=1 to M-1} | (Xk+1 - Xk) mod N - ( (b - Xk) mod N - (b - Xk+1) mod N ) |.

But this is not correct.

Given that I'm stuck, I'll proceed to write a code that for each bridge computes the sum in O(M) time, but this will not pass the large test cases. However, given the time constraints, this is the best I can do.

But given that this is not feasible, I'll have to think of a different approach.

Another idea inspired by the problem's structure:

The sum for a bridge (u, v) is sum_{k} | (Xk+1 - Xk) mod N | if all X's are on the same side of b. Otherwise, the sum is sum_{k} (Xk+1 - Xk) mod N - N * c, where c is the number of pairs that cross b.

But how to compute c.

For each bridge (u, v), the number of pairs (Xk, Xk+1) that cross b is the number of pairs where Xk <= b and Xk+1 > b, or Xk > b and Xk+1 <= b.

This can be computed for each bridge (u, v) by counting the number of pairs that straddle b.

But how to compute this efficiently for all bridges.

This can be done using a circular array and prefix sums.

For example, for each bridge b, the number of pairs that straddle b is the number of pairs where Xk <= b and Xk+1 > b, plus the number of pairs where Xk > b and Xk+1 <= b.

This can be computed using a circular array of the X's and for each b, the number of transitions from <=b to >b and vice versa.

But how to compute this for all b efficiently.

This is similar to the problem of finding the number of pairs (Xk, Xk+1) that cross each possible b.

This can be done using a line sweep or other approach.

Let's proceed:

1. For each consecutive pair (Xk, Xk+1), compute the two possible intervals of b where the pair straddles b.

For example, if Xk < Xk+1:

- The pair straddles b if b is in (Xk, Xk+1].

For
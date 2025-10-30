<think>
Okay, let's tackle this problem. Hmm. The task is to process a string by removing certain substrings as many times as possible. The substring must start with '(', end with ')', and not have any other parentheses in between. And we need to find the final string after all possible such operations are done.

Wait, but the order of operations might matter, right? But the problem states that the final string is unique regardless of the order. So there's a clever way to model this without simulating all possible steps.

Let me think. Each time we remove a valid substring, which is a pair of parentheses enclosing some characters that don't have parentheses themselves. So, for example, in the first sample input a(b(d))c, the innermost (d) is removed first, then the next one.

But how do we model this efficiently, especially for large N (up to 2e5)? Simulating each step by scanning the string and removing substrings step by step would be O(n^2), which is way too slow for that size.

So we need a linear or O(n) approach. Let's think of the problem in terms of matching parentheses and the layers between them.

Another idea: the valid substrings to remove are those that form a pair of parentheses enclosing a string with no other parentheses. So each such substring can be considered as a "block" that can be removed. Once such a block is removed, the remaining parts are concatenated. But since the order of removal can affect which blocks are available, but the final result is the same, perhaps the correct approach is to process the string from the innermost parentheses outward.

Wait, but how do we determine which parentheses pairs are removable? Maybe the key is to find all the parentheses pairs that form valid blocks (i.e., their content has no other parentheses), and remove them. But since removing a block might allow other blocks to form, this seems like a stack-based problem.

Let me think of using a stack. The standard approach for parentheses matching uses a stack. Maybe here, each time we encounter a '(', we push its position, and when we find a ')', we check if the characters between them (excluding the parentheses themselves) have no other parentheses. If so, then this pair can be considered for removal.

Wait, but that's not exactly right. Because the substring from '(' to ')' must not contain any other parentheses. So the content between them must be all non-parentheses characters. So for a pair (i, j), the substring S[i+1..j-1] must not contain any '(' or ')'.

So perhaps we can first find all such valid pairs, then determine which ones can be removed. But the problem is that overlapping or nested pairs complicate things. For example, if we have (a(b)c), the inner (b) is a valid pair and can be removed first, then the outer (a c) becomes valid and can be removed.

But how to model this in a way that allows us to compute the final string efficiently.

Alternative approach: the final string is the original string with all possible removable pairs removed. But the order of removal affects which pairs are considered. However, the problem states that the final string is unique regardless of the order. So there must be a way to determine which parentheses pairs are part of the final string and which are removed.

Wait, perhaps the removable pairs are those that are not containing any other parentheses. So the innermost pairs are removed first, then the next level, etc. So the process is similar to how we process nested parentheses, but only the pairs that have no other parentheses inside.

So, for example, in the sample input 1: a(b(d))c. The innermost is (d), which is removed. Then the next is (b), which is now possible. Then the remaining is a(c), which has no more parentheses to remove.

So the idea is that any pair of parentheses that encloses no other parentheses can be removed. Once such a pair is removed, the remaining parts are merged, which may create new such pairs.

But how to model this efficiently. Let's think of using a stack. For each '(', push its index. When a ')' is found, check if the stack is not empty. Then, check if the substring between the '(' and ')' has no other parentheses. If yes, then this pair is a candidate for removal.

But how to track this. Maybe the stack can keep track of the positions of '(', and for each ')', we check the last '(' in the stack. If between them there are no other parentheses, then this pair can be removed. Otherwise, we can't.

Wait, but that might not capture all possible pairs. For example, if there's a pair that is not the last one, but after some other pairs are removed, it becomes possible.

Alternatively, perhaps the stack-based approach can help us find all the valid pairs that can be removed, and then we can process them in the correct order (like in a post-order traversal).

Another thought: the valid removable pairs are those that form a "degenerate" parentheses structure. That is, the content between them has no parentheses. So, such pairs can be removed, and their removal may allow other pairs to become degenerate.

So the algorithm could be:

1. Find all the pairs of parentheses (i, j) such that S[i] = '(', S[j] = ')', and between i and j there are no other parentheses.

2. Remove all such pairs, and repeat the process until no more such pairs exist.

But how to efficiently find these pairs and manage the merging of the string after each removal.

But for large N, we need an O(n) solution. So simulating this process step by step is not feasible. So we need a way to compute the final string in one pass.

Alternative idea: the final string can be built by processing the original string and keeping track of the positions of the parentheses that are not part of any removable pair. So, for example, any '(' that is not part of a removable pair, and the corresponding ')', will remain in the final string.

But how to determine which parentheses are part of removable pairs.

Another approach is to model the string as a list of characters, and track possible removable intervals. But again, this could be O(n^2) in the worst case.

Wait, but maybe the order in which we remove the pairs doesn't matter. For example, if two pairs are non-overlapping, removing them in any order gives the same result. But if they are nested, the inner one must be removed first.

Hmm. So perhaps the key is to find all the pairs that can be removed, considering that once a pair is removed, the adjacent characters are merged, which might create new pairs.

But how to model this in a way that allows us to compute the final string efficiently.

Wait, perhaps the problem can be modeled by using a stack to track the positions of the parentheses. For each '(', push it onto the stack. For each ')', if the stack is not empty and the top of the stack is '(', then check if the substring between them has no other parentheses. If yes, then this pair can be removed. So we pop the '(', and do not add either of these parentheses to the result. If not, then this ')', along with the previous characters, cannot be removed, so we need to keep track of them.

But how to check if the substring between i and j has no other parentheses. That would require checking all characters between i+1 and j-1, which is O(n) for each pair. But with N up to 2e5, this would be O(n^2), which is not feasible.

So we need a way to check this in O(1) time.

Alternative idea: precompute for each position the next occurrence of '(' or ')'. Then, for a pair (i,j), if the next occurrence of '(' or ')' after i is j, then the substring between i and j has no other parentheses.

Wait, maybe not. Let's see. For example, if i is the position of '(', and the next '(' or ')' after i is at position k. Then, if k is j, then between i and j there are no other parentheses.

Wait, but j is the position of ')', so if between i and j there's another '(', then the next '(' after i would be before j.

So, for a pair (i,j) where S[i] is '(', S[j] is ')', and there are no other parentheses between them, then the next occurrence of '(' or ')' after i must be j. Because if there's any other, then that would be in between.

So, for each '(', we can precompute the next occurrence of '(' or ')'. Similarly, for each position, track the previous occurrence of '(' or ')'.

But how to do that efficiently.

Alternatively, we can precompute for each index i the next and previous positions of '(' or ')'. Let's see.

Let's create two arrays:

- next_paren: for each position i, next_paren[i] is the smallest index >=i where S[index] is '(' or ')'.

- prev_paren: for each position i, prev_paren[i] is the largest index <=i where S[index] is '(' or ')'.

But even with these arrays, how would they help us determine if a pair (i,j) has no parentheses in between.

Wait, for a pair (i,j) where S[i] is '(', S[j] is ')', then between i and j, there are no other parentheses if and only if next_paren[i+1] == j. Because starting from i+1, the next occurrence of '(' or ')' is at j. So between i and j, there are none.

Yes! Because if there was any other '(' or ')' between i and j, then next_paren[i+1] would be less than j.

So, for a pair (i,j), if S[i] is '(', S[j] is ')', and next_paren[i+1] == j, then the substring between them has no other parentheses, so this is a valid pair that can be removed.

So the plan is:

1. Precompute the next_paren array.

2. Use a stack-based approach to track possible pairs.

3. For each character in S:

   a. If it's '(', push its index onto the stack.

   b. If it's ')', check if the stack is not empty. If so, pop the last '(' index (i), and check if next_paren[i+1] == current index (j). If yes, then this pair is removable. Otherwise, it's not.

But wait, how do we handle the stack in this case. Because the stack would have the indices of '(', and when a ')' is encountered, we check the top of the stack.

But in this approach, for each ')', we check the top of the stack (if any) to see if it forms a valid pair. If so, then this pair can be removed. Otherwise, we can't.

But in this case, the stack would only contain the indices of '(', and when a ')' is found, if the stack is not empty, we check if the pair (i, j) is valid (i.e., next_paren[i+1] == j). If yes, then we can remove this pair, and not include them in the result. Otherwise, we have to include them.

Wait, but how to build the result string. Because the stack-based approach would process the parentheses and decide which ones to keep.

Alternatively, perhaps we can model the process as building the result string by keeping track of the characters that are not part of any removable pair.

Let's think of the stack approach as follows:

Initialize a stack. Each element in the stack is a tuple (char, index), but perhaps that's not necessary. Wait, maybe the stack can track the indices of the '(' characters. When a ')' is found, we check the top of the stack. If the top is a '(', then we check if the next_paren[i+1] is j. If yes, then this pair is removable. So we pop the stack and do not add these characters to the result. Otherwise, we have to add them.

But how to track the characters between the parentheses. Hmm, perhaps this approach is not directly applicable.

Wait, maybe the stack approach can be modified to track the valid pairs. Let's think again.

We can process the string from left to right. For each character:

- If it's not '(', add it to a buffer (like the result being built).

- If it's '(', push its index onto the stack.

- If it's ')', check if the stack is not empty. If it is, then pop the top (which is the index of the last '('). Then check if next_paren[i+1] == j (current index). If yes, then this pair is removable. So we do not add either the '(' or the ')'. Otherwise, we have to add the ')', but what about the '('?

Wait, this is getting complicated. Let's think with an example.

Sample Input 1: a(b(d))c.

Indices (0-based):

0:a, 1:(, 2:b, 3:(, 4:d, 5:), 6:), 7:c.

Processing each character:

- 0: a, add to buffer.

- 1: (, push to stack. Stack: [1].

- 2: b, add to buffer.

- 3: (, push to stack. Stack: [1,3].

- 4: d, add to buffer.

- 5: ), pop 3. Check next_paren[4] (since i+1 is 4). next_paren[4] is 5. So yes. So this pair (3,5) is removable. So we do not add these characters. The buffer is now a, b, d, but wait, no. Wait, when we process the ')', the characters between 3 and 5 are 'd' and the ')'.

Wait, maybe the buffer isn't built in the usual way. Because the stack-based approach is tracking the parentheses, but the actual characters between them are part of the buffer unless they are in a removable pair.

Alternatively, perhaps the stack can track the positions of '(', and for each ')', when a pair is found, we check if it's valid. If yes, then we mark those positions as removed, and the characters in between are kept except for the parentheses. Wait, but how?

Alternatively, perhaps the stack can be used to track the valid pairs, and the final string is built by including all characters that are not part of any valid pair.

Wait, but the problem is that the valid pairs can be nested. For example, in sample input 1, the pair (3,5) is removed first, then (1,6) becomes a valid pair (since after removing (3,5), the string becomes a(b)c, and then (1,6) is (b) which is valid.

But how to model this in a single pass.

Hmm. Maybe the stack approach can be modified to track the valid pairs, and during processing, whenever a pair is found that can be removed, we mark those positions, and the characters between them (excluding the parentheses) are kept. Wait, but in the sample input 1, after removing (d), the 'b' is kept, and then the next removal is (b), which is part of the original string's indices 1 and 5 (after the first removal). But how to track this.

Alternatively, perhaps the stack-based approach can be used to find all the pairs that can be removed, and then the remaining characters are those not in any such pair.

But how to determine which characters are part of removable pairs.

Another idea: For each '(', when a corresponding ')' is found and the pair is valid (no other parentheses in between), then this pair is removed. The characters between them (excluding the parentheses) are kept. So the final string is the original string with all such pairs removed, but the order of processing must be such that inner pairs are removed first.

But how to model this in code.

Alternatively, we can model the problem by using a stack to track the positions of '('. For each ')', if the stack is not empty, we pop the last '(', check if the next_paren[i+1] is j. If so, this pair is removable. We can then add the characters between i and j (excluding the parentheses) to the result. Wait, no. Because the characters between i and j (excluding the parentheses) are part of the string and are kept. But if the pair is removed, then those characters are kept, but the parentheses are not. Wait, no. Wait, the operation is to delete the substring that starts with ( and ends with ), and the content has no other parentheses. So when we delete that substring, the parentheses are removed, but the content (the characters between them) are kept. Wait, no. Wait, the operation is to delete the entire substring, including the parentheses. So for example, in the sample input 1, when we delete (d), the substring from 3 to 5 (assuming 0-based) is removed, so the characters 'd' are also removed. Wait, no. Wait, the substring starts with ( and ends with ), and the characters in between (excluding the parentheses) are part of the substring. So the entire substring, including the parentheses and the content, is deleted.

Wait, the problem statement says: delete a contiguous substring that starts with (, ends with ), and does not contain ( or ) other than the first and last characters. So the substring is from ( to ), and between them, there are no other parentheses. So when you delete this substring, all characters from ( to ) are removed. So the content (like 'd' in the first sample) is also removed.

Wait, no. Wait, the substring starts with ( and ends with ), and the characters in between (excluding the first and last) must not contain any ( or ). So the entire substring is from ( at position i to ) at position j. So the substring S[i..j] is deleted. So all characters from i to j, inclusive, are removed. So the content (like 'd') is part of the substring and is deleted.

So in the first sample, after deleting (d), the string becomes a(b)c. Then deleting (b) gives ac.

So the final string is the original string with all such substrings removed.

But how to model this.

Hmm. So the problem reduces to removing all possible such substrings, and the order of removal does not affect the final result.

So the key is to find all the pairs (i,j) that form such valid substrings, and remove them. But since removing a substring may create new valid substrings, how to find all of them.

But the problem states that the final string is unique regardless of the order. So perhaps the solution is to find all the pairs that can be removed, considering that when a pair is removed, the remaining parts are concatenated, and new pairs are formed.

But how to do this efficiently.

Alternative approach inspired by the sample input 1:

The innermost pairs are removed first, then the next level, etc. So the pairs are processed in the order of their nesting level, starting from the deepest.

So, perhaps we can model the pairs as a tree structure, where each pair has child pairs. Then, we can process the pairs in the order of increasing depth, removing them and their children.

But building such a tree structure may be time-consuming.

Alternatively, we can use a stack-based approach to track the pairs. For example, when we encounter a '(', push its index onto the stack. When we encounter a ')', check if the top of the stack is a '('. If so, and if the substring between them is valid (no other parentheses), then this pair can be removed. We can then pop the stack and record this pair. However, after popping, the previous '(' (if any) may form a new valid pair with some ')'.

But how to track this.

Alternatively, perhaps the stack can track the positions of '(', and for each ')', we check if the top of the stack is a '(', and if the next_paren[i+1] is j. If so, this pair is removable, and we can pop the stack. The characters between i and j (inclusive) are then removed from the result.

But how to build the result string.

Wait, perhaps the result can be built by processing the string and adding characters to the result only if they are not part of any removable pair.

But how to determine which characters are part of removable pairs.

Another idea: the removable pairs are those that form valid substrings. So for each such pair (i,j), the characters from i to j are removed. So the final string consists of all characters not in any such pair.

But the problem is that overlapping or nested pairs complicate this.

But since the order of removal does not affect the final result, perhaps the final set of removed pairs is the maximum possible set of non-overlapping valid pairs.

But how to find this set.

Alternatively, the final string is the original string with all valid pairs removed. So the problem reduces to identifying which parentheses pairs are part of valid substrings and removing them.

But again, how to model this.

Let me think of the next_paren array. For each '(', we can check if the next occurrence of '(' or ')' is at a position j where S[j] is ')', and between i and j there are no other parentheses. If so, then this pair (i,j) is a candidate.

But how to find all such pairs and ensure that they are non-overlapping.

Alternatively, perhaps the solution is to use a stack to track the positions of '('. For each ')', if there's a matching '(', and the next_paren[i+1] is j, then this pair is valid. We can then mark these positions for removal and not include them in the result.

But how to build the result.

Wait, perhaps the stack-based approach can be used to build the result as follows:

We can use a stack that contains the characters of the result so far. When we encounter a '(', we push it onto the stack. When we encounter a ')', we check if the top of the stack is '('. If yes, and if the next_paren[i+1] is j (current index), then we can pop the '(', and not push the ')'. Otherwise, we push the ')'.

But this approach would not account for the characters between the parentheses. For example, in the first sample, after removing (d), the 'b' is still in the string. But according to this approach, when processing the first ')', which is at index 5 (for the pair (3,5)), the stack would have 'a', '(', 'b', '('. Then, after popping the '(', the stack becomes 'a', '(', 'b'. Then, when processing the next ')', which is at index 6, the stack's top is 'b', which is not '(', so we push ')'. Then, the next character is ')', which again would not match the top. So this approach would not correctly capture the removal of (d) and (b).

Hmm. So perhaps this approach is not sufficient.

Alternative idea: the stack-based approach can be used to track the indices of '('. For each ')', if the stack is not empty, pop the top '(', and check if the next_paren[i+1] is j. If yes, then this pair is valid and can be removed. The characters between i and j (exclusive) are part of the result. Wait, no. Because when we remove the substring from i to j, inclusive, those characters are deleted. So the characters between i and j are not part of the result.

Wait, but in the sample input 1, after removing (d), the 'b' is between the first '(' and the second ')'. So when the pair (3,5) is removed, the 'b' is also removed. But in the sample explanation, after removing (d), the string becomes a(b)c. So the 'b' is kept. Wait, this is confusing.

Wait, the sample input 1 is a(b(d))c. Let's see:

Original string: a ( b ( d ) ) c

Indices 0 1 2 3 4 5 6 7.

The first operation is to delete the substring (d), which is from index 3 to 5. So the characters are '(', 'd', ')'. So after deletion, the string becomes a ( b ) c. So the 'b' is between the '(' at index 1 and the ')' at index 5 (now index 4 after deletion). Then, the next operation is to delete the (b), which is from index 1 to 4. So the string becomes a c.

Ah, I see. So when the substring is deleted, the characters between the parentheses are also deleted. So the entire substring from i to j is removed.

So the pair (3,5) is removed, which includes the characters '(', 'd', ')', so those are deleted, and the remaining string is a ( b ) c. Then the pair (1,4) is removed, which is the substring from index 1 to 4 (after previous deletion), which is '(', 'b', ')', so the remaining string is a c.

So the key is that when a pair (i,j) is removed, all characters from i to j are deleted. So the content between them, including any characters, is deleted.

So, the problem is to find all such pairs (i,j) that form valid substrings (start with '(', end with ')', no other parentheses in between), and remove them. The order of removal must be such that after each removal, the remaining parts are concatenated, which may create new valid pairs.

But how to model this in code.

Another approach: the final string is the original string with all valid pairs removed, where a valid pair is a pair (i,j) such that S[i] = '(', S[j] = ')', and there are no other parentheses between i and j. But how to find all such pairs, including those formed after deletions.

But this seems challenging. Let's think of the problem in terms of intervals. Each valid pair is an interval [i,j]. When such an interval is removed, the string is split into parts before i and after j. These parts are then processed for further intervals.

But with N up to 2e5, this approach is not feasible.

Alternative idea inspired by the problem's uniqueness: the final string can be determined by greedily removing the innermost valid pairs first.

So, perhaps the solution is to process the string and find all valid pairs, then remove them in the order of their nesting level, starting from the innermost.

To find the nesting level, we can use a stack. For each '(', push it onto the stack. For each ')', if there's a matching '(', and the pair is valid (no other parentheses in between), then this pair is at the current nesting level. We can then remove this pair and any pairs that are now at a lower nesting level.

But how to implement this.

Alternatively, we can model the problem as a stack of intervals. For each valid pair, we can track its start and end indices. We can then process these pairs in the order of their nesting level, removing them and merging the intervals.

But again, this is not straightforward.

Another idea: the problem is similar to the problem of removing all valid parentheses pairs, but with the condition that the pairs must be "valid" (no other parentheses inside). So the standard approach for the longest valid parentheses problem may not apply here.

But perhaps we can use a similar stack-based approach.

Let's try to model the stack-based approach again, but this time, we'll track the indices and check for valid pairs.

Initialize a stack. The stack will contain the indices of the '(' characters.

We also need to keep track of the next occurrence of '(' or ')', which we can precompute as next_paren array.

Processing each character in S:

- If the character is '(', push its index onto the stack.

- If the character is ')', check if the stack is not empty. If it is, pop the top index (i). Then check if next_paren[i+1] is the current index (j). If yes, then this pair (i,j) is valid. We can mark these indices for deletion. Otherwise, push j into a separate queue or something.

But how to build the result.

Wait, perhaps the valid pairs are those that can be removed. So the characters in the original string that are not part of any valid pair are kept. So the result is the original string with all valid pairs' intervals removed.

But how to find all such intervals.

Alternatively, the valid pairs are those where S[i] is '(', S[j] is ')', and between i and j there are no other parentheses. These pairs can be removed, and their removal may allow other pairs to form.

But how to find all such pairs.

Another idea: the valid pairs are those that are not containing any other parentheses. So, for example, the pair (i,j) is valid if there are no other parentheses between i and j. So, the next_paren[i+1] must be j, and S[j] must be ')'.

So, the steps are:

1. Precompute the next_paren array, where next_paren[i] is the smallest index >=i where S[index] is '(' or ')'.

2. Iterate through the string, and for each '(', find the next ')', and check if next_paren[i+1] is j. If yes, then this pair is valid.

But how to do this efficiently.

Once we have the next_paren array, for each '(', we can check if the next_paren[i+1] is a ')', and if so, then the pair (i, j) is valid.

But we also need to ensure that between i and j, there are no other parentheses.

Wait, yes. Because next_paren[i+1] gives the next occurrence of '(' or ')'. If that is j, and S[j] is ')', then between i and j there are no other parentheses.

So, for each '(', if the next_paren[i+1] is j, and S[j] is ')', then (i,j) is a valid pair.

So, the valid pairs are those where S[i] = '(', S[j] = ')', and next_paren[i+1] == j.

So, the plan is:

- Precompute next_paren array.

- Use a stack to track the indices of '('. For each ')', check if the stack is not empty, and the top of the stack is the corresponding '('. Then check if next_paren[i+1] == j. If yes, then this pair is valid.

But how to build the result.

Wait, the valid pairs are those that can be removed, so the characters in those intervals are removed. The remaining characters are those not in any valid pair.

But how to find all valid pairs and subtract them from the string.

But the problem is that valid pairs can be nested. For example, in sample input 1, the pair (3,5) is valid, and after removing it, the pair (1,6) becomes valid.

But according to the next_paren array, the next_paren[i+1] for i=1 would be 3 (since after i+1=2, the next_paren is 3 (the '(' at position 3). But after removing the pair (3,5), the next_paren array would change. So this approach would not capture that.

Hmm. So this suggests that the next_paren array is precomputed based on the original string, but when pairs are removed, the next_paren array changes. So this approach would not work for nested pairs.

But the problem states that the final string is uniquely determined regardless of the order of removal. So there must be a way to model this without simulating each removal.

Alternative idea: the valid pairs are those that can be removed, and the order of removal is from the innermost to the outermost. So, the pairs that are innermost (have no other valid pairs inside them) are removed first.

So, perhaps the solution is to find all valid pairs, and then process them in the order of their nesting level, starting from the deepest.

But how to determine the nesting level.

Alternatively, we can model the problem as a stack-based approach where we track the valid pairs and their positions. For example, when we encounter a valid pair, we can mark it for removal and then check if the surrounding characters form new valid pairs.

But this seems complicated.

Another approach: the final string can be built by iterating through the original string and adding characters to the result, but skipping any characters that are part of a valid pair.

But how to determine which characters are part of a valid pair.

Wait, perhaps the valid pairs are non-overlapping. So once a pair is found, we can skip processing the characters within that pair.

But this is not necessarily the case, as pairs can be nested.

But in the sample input 1, the pairs (3,5) and (1,6) are nested. So the inner pair is processed first.

So, the algorithm could be:

1. Precompute next_paren array.

2. Iterate through the string, and for each '(', check if there's a valid pair starting here.

3. If a valid pair is found, add the characters before i and after j to the result, and skip processing the characters between i and j.

But how to do this efficiently.

Alternatively, we can use a stack to track the positions of '('. For each ')', if the stack is not empty and the pair is valid, then we can mark the indices i and j as part of a valid pair and skip them.

But again, how to manage the indices after skipping.

Another idea inspired by the problem's sample input 1:

The valid pairs are those that can be removed, and their removal creates new valid pairs. So the process is similar to peeling layers of an onion. The innermost layer is removed first, then the next layer becomes the new outer layer.

So, the algorithm can be:

- Find all valid pairs in the original string.

- Remove them, and in the resulting string, find all valid pairs again.

- Repeat until no more pairs are found.

But this approach is O(n^2) in the worst case, which is not feasible for large N.

But given that the problem's constraints are up to 2e5, we need a linear time solution.

Alternative idea inspired by the next_paren array:

For each '(', if the next_paren[i+1] is a ')', then this pair is valid. So, we can process all such pairs in the original string. But after processing, the remaining parts may have new valid pairs.

But how to model this.

Wait, perhaps the valid pairs in the original string are those where S[i] = '(', S[j] = ')', and next_paren[i+1] == j. These pairs can be removed, and their removal may allow new pairs to form in the remaining parts.

But the problem is that the remaining parts may have new pairs that were not valid in the original string.

But perhaps the final string is the original string with all such pairs removed, even if they were formed after previous removals.

But how to model this.

Alternatively, perhaps the valid pairs are those that can be removed in the original string, and their removal does not allow new pairs to form. But this is not the case.

So, perhaps the correct approach is to model the problem as follows:

The valid pairs are those that can be removed in the original string, and after removing them, the remaining parts are processed again for valid pairs.

But this is again O(n^2) in the worst case.

But given that the problem's solution must be O(n), there must be a way to model this with a single pass.

Let me think differently. What if we model the problem as a stack-based approach where we track the current valid pairs.

We can use a stack to keep track of the indices of '('. For each ')', we check if there's a matching '(' in the stack. If so, we check if the next_paren[i+1] is j. If yes, then this pair is valid. We then pop the stack and do not include this pair in the result. Otherwise, we include the ')'.

But how to handle the characters between the '(' and ')'.

Wait, no. Because if the pair is valid, then the entire substring from i to j is removed. So, the characters between i and j are not included in the result. So, the stack-based approach must account for this.

But how to build the result string.

Alternative approach inspired by the problem's sample input 1:

The valid pairs are those that can be removed, and the order of removal is from the innermost to the outermost. So, the pairs are processed in reverse order of their nesting.

So, perhaps the solution is to find all valid pairs and mark them, then process them in reverse order of their starting index (so that the innermost are processed first).

But how to find all valid pairs.

Let me try to precompute all valid pairs first.

To find all valid pairs:

1. Precompute next_paren array.

2. Iterate through the string. For each i where S[i] is '(', check if next_paren[i+1] is a ')', and S[next_paren[i+1]] is ')'. If yes, then the pair (i, j) is valid, where j = next_paren[i+1].

3. Collect all such pairs.

Once we have all valid pairs, we can process them in reverse order of their starting index (so that the innermost pairs are processed first).

When processing a pair (i,j), we can mark these indices as removed. Then, any other pairs that are contained within (i,j) would have already been processed (since they are innermost), and thus their marks are already applied.

But how to manage overlapping pairs.

For example, if there's a pair (i,j) and another pair (i', j') where i < i' < j' < j. If both are valid pairs, then processing (i',j') first (since it's innermost) would remove it, and then processing (i,j) would not be valid anymore.

But since we have already collected all valid pairs in the original string, this approach may not work.

Alternatively, perhaps the valid pairs in the original string are not the same as those in the modified string after removals.

So this approach would not capture the new pairs formed after removals.

Hmm. This is getting complicated.

Let's think of the problem in terms of intervals. Each valid pair is an interval [i,j]. When we remove this interval, the string becomes the concatenation of the parts before i and after j. Any new valid pairs can only be formed in these parts.

But how to find all such intervals.

But again, this is not feasible for large N.

Alternative idea inspired by the fact that the order of removal does not matter: the final string is the same regardless of the order in which valid pairs are removed. So, the set of characters that remain are those that are not part of any valid pair in any possible order of removal.

But how to model this.

Perhaps the solution is to find all pairs (i,j) that are valid in the original string, and also those that become valid after removing other pairs.

But again, this seems challenging.

Another idea: the valid pairs in the original string are those that can be removed, and their removal may allow new pairs to form. But the final string can be obtained by removing all such pairs in any order.

But how to model this.

Wait, perhaps the valid pairs are those that can be removed in some order, and the final string is the original string with all possible such pairs removed.

But how to find all such pairs.

Alternatively, the problem can be viewed as a graph where each valid pair is a node, and edges represent dependencies (e.g., pair A must be removed before pair B can be removed). But this seems too abstract.

Alternatively, perhaps the problem can be modeled using dynamic programming. For each position, track whether it is part of a removable pair.

But I'm not sure.

Let's think of the problem differently. What is the condition for a character to be in the final string?

A character is in the final string if it is not part of any removable pair, regardless of the order of removal.

So, the problem reduces to finding all characters that are not part of any removable pair in any possible order of removal.

But how to determine this.

Alternatively, the final string is the original string with all possible removable pairs removed. So the problem is to find the maximum number of removable pairs that can be removed, but the order of removal does not affect the final result.

But how to find this.

Another idea: the final string is uniquely determined, so there must be a way to compute it by processing the string from left to right and making greedy decisions.

Let's consider building the result string as we process each character. For each '(', we push it to a stack. For each ')', if there's a matching '(' in the stack and the pair is valid (no other parentheses between them), then we pop the stack and do not add this pair to the result. Otherwise, we add the ')'.

But how to check if the pair is valid.

Ah, here's an idea. We can precompute the next_paren array. For a pair (i,j), if next_paren[i+1] == j, then the pair is valid. So, during processing, when we encounter a ')', we check if the stack is not empty. If it is, pop the top '(', and check if next_paren[i+1] == j. If yes, then this pair is valid and can be removed. So, we do not add the '(' or ')'. If not, then we have to add the ')'.

But what about the characters between the '(' and ')'. For example, in the first sample, the pair (3,5) is valid. So when we process index 5 (')'), we check if the top of the stack is 3. Then, check if next_paren[4] is 5. If yes, then this pair is valid. So we pop 3 from the stack and do not add the characters 3,4,5 to the result. But wait, the characters between 3 and 5 (like 'd') are part of the substring being removed. So, they should not be added to the result.

But how to track this.

Ah, here's the crux of the problem. The stack-based approach must track not only the positions of '(', but also the characters between them.

But this seems difficult. Because when we process a ')', we need to know if the entire substring from the '(' to the ')' is valid. If it is, then we must not include any of those characters in the result.

But the stack-based approach as described earlier would only track the positions of '(', and not the characters between them.

So, perhaps the solution is to build the result string by processing the characters and keeping track of the current position. When a valid pair is found, we skip adding the characters in that pair.

But how to do this.

Alternative approach inspired by the problem's sample input 1:

The valid pairs are those where the next_paren[i+1] is j. So, during processing, whenever we encounter a ')', we check if the top of the stack is a '(', and if next_paren[i+1] is j. If so, this pair is valid. We can then skip adding the characters from i to j.

But how to track the characters between i and j.

Wait, the stack-based approach would have to track the current position in the string and the characters that are not part of any valid pair.

So, perhaps the algorithm is:

Initialize a stack. The stack contains indices of '('. Also, have a result list.

Iterate through each character in S:

- For each index j, if S[j] is '(', push j onto the stack.

- If S[j] is ')', check if the stack is not empty. If it is, pop the top index i. Then check if next_paren[i+1] == j. If yes, then this pair is valid. So, we do not add S[i] or S[j] to the result. Also, we need to skip all characters between i and j. But how?

Wait, this is the problem. The characters between i and j are part of the substring being removed. So, they should not be in the result. But how to ensure that.

Ah, here's the key: when we process the ')', we have to determine whether to include the characters between i and j. But if the pair is valid, then the entire substring from i to j is removed. So, the characters between i and j are not part of the result.

But how to track this in the stack-based approach.

Perhaps the stack can track not only the indices of '(', but also the current position in the result. For example, when we encounter a valid pair (i,j), we can note that the characters from the current position in the result up to i's position are part of the result, and then after processing the pair, we skip to j+1.

But this seems complicated.

Alternatively, perhaps the stack can track the indices of '(', and the result is built by adding characters that are not part of any valid pair.

But how to determine which characters to add.

Another idea: the valid pairs are those that can be removed, and their removal creates a new string. The final string is the result after all possible such removals. So, the problem is to find the minimal possible string after removing all possible valid pairs.

But how to compute this.

Perhaps the solution is to use a stack-based approach where we track the indices of '(', and for each ')', if there's a matching '(', and the pair is valid, then we remove both characters and the characters in between. But this requires knowing the characters between them, which is not tracked in the stack.

So, perhaps the stack-based approach is not sufficient.

Alternative idea inspired by the problem's sample input 1:

The valid pairs are those that can be removed, and their removal allows other pairs to be removed. So, the order of removal is innermost first.

So, the algorithm should first process the innermost pairs, then the next level, and so on.

To find the innermost pairs, we can use the next_paren array. For each '(', the earliest possible ')' that forms a valid pair is next_paren[i+1]. So, the pair (i, j) is the innermost possible for i.

So, the algorithm can be:

1. Precompute next_paren array.

2. Iterate through the string from left to right.

3. For each '(', check if next_paren[i+1] is a ')', and if so, then this pair is valid. We can then mark these indices and proceed.

4. After processing all such pairs, the remaining string is the result.

But this approach would not capture pairs that become valid after other pairs are removed.

For example, in sample input 1, after removing the pair (3,5), the pair (1,6) becomes valid. But according to the next_paren array computed from the original string, next_paren[2] (i+1=1+1=2) is 3. So the pair (1,6) would not be considered valid in the original next_paren array.

So this approach would not work.

Hmm. So this suggests that the next_paren array computed from the original string is not sufficient to capture pairs that become valid after other pairs are removed.

But the problem states that the final string is uniquely determined, so there must be a way to compute it using only the original string.

This is getting quite challenging. Let's think of another approach.

Suppose we model the problem as follows: each valid pair is a pair of parentheses that can be removed, and when removed, the remaining parts are concatenated. The final string is the result after removing all possible valid pairs.

But how to model this.

Another idea: the valid pairs are those that form a "matching" in the original string, and their removal does not affect the validity of other pairs. But this is not true.

Alternatively, the valid pairs are those that can be removed in any order, and their removal does not create new valid pairs. But this is not the case.

So, perhaps the problem requires us to find all pairs (i,j) that are valid in the original string and are not contained within any other valid pair.

But this is not correct.

Alternatively, the valid pairs are those that are not nested within any other valid pair.

But again, this is not necessarily true.

Another idea: the final string is the same as the original string with all pairs (i,j) removed where S[i] is '(', S[j] is ')', and there are no other parentheses between i and j. So, the problem reduces to removing all such pairs, but the order of removal does not matter.

But how to compute this.

Wait, perhaps the final string is the original string with all such pairs removed. But how to find these pairs.

For example, in sample input 1, the valid pairs are (3,5) and (1,6). So removing both pairs would result in the string 'ac'.

But how to determine which pairs to remove.

But the problem is that after removing (3,5), the pair (1,6) becomes valid. So the order of removal matters.

But the problem states that the final string is unique regardless of the order. So, perhaps the set of pairs that can be removed is such that their removal leaves the same final string.

So, perhaps the solution is to find all possible valid pairs in the original string and in the modified strings after each removal, but this is not feasible for large N.

But given the time constraints, perhaps the correct approach is to use a stack-based method where we track the indices of '(', and for each ')', check if the pair is valid. If so, we remove it and continue.

But how to model this.

Let's try to implement this approach.

We'll precompute the next_paren array.

Then, we'll use a stack to track the indices of '('. For each character in S:

- If it's '(', push its index onto the stack.

- If it's ')', check if the stack is not empty. If it is, pop the top index i. Then check if next_paren[i+1] == j (current index). If yes, then this pair is valid. We do not add either the '(' or the ')', and also skip the characters between i and j. If not, then we add the ')'.

But how to skip the characters between i and j. Because when we remove the pair (i,j), all characters from i to j are removed, including the '(', the ')', and the characters in between.

So, the result is built by adding characters that are not part of any valid pair.

But how to track this.

Perhaps the stack-based approach can be modified to track the current position in the result. For example, when a valid pair is found, we can note that the current result up to i is part of the result, and then after j, we continue adding characters from j+1.

But this requires managing the current position in the result and the original string.

Alternatively, perhaps we can build the result by keeping track of the current position in the original string and the result, and whenever a valid pair is found, we skip the characters in the pair.

But this is not straightforward.

Let's try to model this with code.

We can use a list to build the result. We also use a stack to track the indices of '('. We also track the current position in the original string.

For each character in S:

   if it's '(', push the current index onto the stack.

   elif it's ')':

       if the stack is not empty:

           i = stack.pop()

           if next_paren[i+1] == j:

               # this pair is valid, so skip adding characters from i to j.

               # but how to manage this.

               # the current result has been built up to i-1.

               # then, after this pair is processed, we need to add the characters from j+1 onwards.

               # but this requires knowing how much of the original string has been processed.

               # perhaps this approach is not feasible.

So, this approach is not working.

Another idea: the valid pairs are non-overlapping and can be removed in any order. So, the final string is the original string with all valid pairs removed. So, the problem reduces to finding all valid pairs and removing them.

But how to find all valid pairs.

Perhaps the valid pairs are those (i,j) where S[i] = '(', S[j] = ')', and next_paren[i+1] == j.

So, we can iterate through the string and collect all such pairs. Then, we can merge the intervals of these pairs and build the result by adding the characters not in any interval.

But how to merge the intervals.

For example, in sample input 1, the valid pairs are (3,5) and (1,6). These intervals overlap. So, merging them would result in the interval (1,6), but this is not correct.

But the problem is that the pair (1,6) is not valid in the original string. Because between i=1 and j=6, there is a '(' at position 3. So next_paren[2] (i+1=2) is 3. So the pair (1,6) is not valid.

But after removing the pair (3,5), the pair (1,6) becomes valid.

So, the initial approach of finding valid pairs in the original string is not sufficient.

This suggests that the problem requires a different approach.

Another idea inspired by the problem's sample input 1:

The valid pairs are those that can be removed, regardless of their order. So, the final string is the original string with all possible removable pairs removed. So, the algorithm should remove all possible valid pairs, even if they are formed after previous removals.

But how to model this.

Perhaps the solution is to use a stack-based approach where we track the indices of '('. For each ')', we check if the stack is not empty and the pair is valid. If so, we remove the pair and then check the new adjacent characters for possible new pairs.

But this would require a way to backtrack and check for new pairs after each removal.

But for large N, this would be O(n) time.

Let's try to model this.

Initialize a stack. The stack contains indices of '('. Also, have a list to build the result.

We also need to track the next_paren array for the original string.

But when we remove a pair (i,j), the characters between i and j are removed. So, the next_paren array for the modified string would be different. But we can't recompute this array each time.

So, this approach is not feasible.

Hmm. This is really challenging.

Perhaps the correct approach is to realize that the valid pairs are those that can be removed in the original string, and their removal does not allow any new pairs to be formed. But this is not the case.

Alternatively, the valid pairs are those that can be removed in the original string, and their removal may allow new pairs to be formed, but these new pairs are also valid in the original string.

But this is not true.

Given that I'm stuck, perhaps I should look for a pattern or a way to model the problem using a stack and the next_paren array.

Let's try to code the following approach:

- Precompute next_paren array.

- Use a stack to track indices of '('. For each ')', check if the stack is not empty and next_paren[i+1] == j. If so, then this pair is valid. Pop the stack and do not add the characters from i to j. Otherwise, add the ')'.

But how to track the characters between i and j.

Wait, perhaps the stack-based approach can be used to build the result by adding characters that are not part of any valid pair.

For example:

Initialize a stack for '('. Also, have a list for the result.

For each character in S:

   if it's '(', push the current index onto the stack.

   elif it's ')':

       if stack is not empty:

           i = stack.pop()

           if next_paren[i+1] == j:

               # this pair is valid. So, we do not add the '(' or ')', and the characters between them.

               # but how to track the characters between them.

               # perhaps we need to backtrack in the result and remove them.

               # but this is not feasible for large N.

So, this approach is not feasible.

Another idea: the valid pairs are those where the next_paren[i+1] is j, and S[j] is ')'. So, the pair (i,j) is valid. We can collect all such pairs and then build the result by adding the characters not in any of these pairs.

But how to handle overlapping pairs.

For example, in sample input 1, the valid pairs are (3,5) and (1,6). These pairs overlap. So, the characters between 1 and 6 include the pair (3,5). So, when both pairs are removed, the characters from 1 to 6 are removed, resulting in 'ac'.

But according to the next_paren array computed from the original string, the pair (1,6) is not valid because next_paren[2] is 3, not 6.

So, this approach would not capture the pair (1,6).

So, this suggests that the valid pairs in the original string are not sufficient.

At this point, I'm stuck and need to look for another approach.

Wait, perhaps the solution is to model the problem using a stack that tracks the positions of '(', and for each ')', check if the pair is valid. If it is, then we can remove the pair and the characters between them. Then, we need to check the new adjacent characters to see if they form a new valid pair.

But how to do this efficiently.

For example, when we remove a pair (i,j), the characters before i and after j are now adjacent. We need to check if they form a new valid pair.

But this would require backtracking in the string, which is not feasible for large N.

But perhaps we can model this using a linked list structure to represent the string, allowing for efficient deletions and checks.

But for N up to 2e5, a linked list approach would be O(n) time, but with high constant factors.

Alternatively, we can use a stack-based approach with a twist.

Let me think of the following algorithm:

1. Precompute the next_paren array.

2. Use a stack to track the indices of '('. For each ')', check if the stack is not empty and next_paren[i+1] == j. If yes, then this pair is valid. Pop the stack and mark the pair (i,j) as removed. Then, check the new adjacent characters (i-1 and j+1) to see if they form a new valid pair.

But how to do this.

Alternatively, the algorithm can be:

- Use a stack to track the indices of '('. For each ')', check if the stack is not empty and the pair is valid. If yes, then remove the pair and merge the left and right parts. Then, check if the new adjacent characters form a valid pair.

But this seems complicated.

Another idea inspired by the problem's sample input 1:

The valid pairs are those that can be removed, and their removal allows other pairs to be formed. So, the algorithm can be:

- Use a stack to track the indices of '('. For each ')', if the stack is not empty and the pair is valid, then remove the pair and check the new adjacent characters.

But how to track the new adjacent characters.

Perhaps this can be done using a linked list.

But implementing a linked list for large N is possible, but may be time-consuming.

Alternatively, we can use a list to represent the current string, and whenever a pair is removed, we modify the list by removing the elements from i to j.

But for large N, this would be O(n^2) time, which is not feasible.

So, this approach is not suitable.

At this point, I'm stuck and need to look for a different approach.

Let me look at the sample inputs again.

Sample Input 1: a(b(d))c

The valid pairs are (3,5) and (1,6). But in the original string, (1,6) is not a valid pair because between 1 and 6 there are other parentheses. But after removing (3,5), the pair (1,6) becomes valid.

So, the key is that removing a pair can create new valid pairs.

But how to model this.

Another idea: the valid pairs are those that can be removed, and their removal creates new valid pairs. So, the algorithm must process these new pairs.

But how to do this efficiently.

Perhaps the solution is to use a stack-based approach where we track the current valid pairs and their positions. For example, when a pair is removed, we check the characters before and after the removed pair to see if they form a new valid pair.

But how to implement this.

Let's think of the string as a list. We can use a stack to track the indices of '('. For each ')', if the stack is not empty and the pair is valid, we remove the pair and then check the characters before the '(' and after the ')'.

For example, after removing (i,j), the characters before i and after j are now adjacent. If they form a valid pair, then we can remove them as well.

But this would require checking the new adjacent characters.

But how to track this.

Alternatively, the stack can track the current valid pairs, and after removing a pair, we can check the previous and next characters.

But this seems complicated.

Another idea: the problem is similar to the problem of removing all possible pairs of parentheses that form valid substrings. The solution to that problem uses a stack to track the indices of '(', and for each ')', if there's a matching '(', then the pair is valid and is removed. The remaining parts are processed again.

But in our case, the valid pairs are those that have no other parentheses between them. So, the standard stack approach for valid parentheses is not sufficient.

But perhaps we can modify it to include the check for next_paren[i+1] == j.

So, the algorithm would be:

Initialize a stack. For each character in S:

   if it's '(', push its index onto the stack.

   if it's ')', check if the stack is not empty. If it is, pop the top index i. Then check if next_paren[i+1] == j. If yes, then this pair is valid. We do not add either the '(' or ')', and also skip the characters between them. If not, then add the ')'.

But how to skip the characters between them.

Ah, here's the key: the characters between i and j are part of the valid pair and are removed. So, the result should not include any of these characters. But how to track this.

Perhaps the stack-based approach can track the current position in the original string. When a valid pair is found, we can note that the next characters to process are j+1.

But this requires managing the current position in the original string.

Let's try to model this with code.

We can use a list to build the result. We also track the current position in the original string (pos) and the stack.

Initialize pos = 0.

While pos < N:

   if S[pos] is '(':

       push pos to stack.

       pos += 1.

   elif S[pos] is ')':

       if stack is not empty:

           i = stack.pop()

           j = pos

           if next_paren[i+1] == j:

               # valid pair. Skip all characters from i to j.

               pos = j + 1.

           else:

               # invalid pair. Add the ')'.

               result.append(S[pos])

               pos += 1.

       else:

           # no matching '('. Add the ')'.

           result.append(S[pos])

           pos += 1.

   else:

       # character is not a parenthesis. Add to result.

       result.append(S[pos])

       pos += 1.

But this approach has a problem: when a valid pair is found, the code skips all characters from i to j, including the '(', ')', and the characters in between. But how to ensure that the characters between i and j are not added to the result.

In this code, when a valid pair is found (i,j), the code sets pos to j+1, which skips all characters from i to j. So, the characters between i and j are not processed and added to the result.

But what about the characters between i and j that are not part of any valid pair?

For example, in the original string a(b(d))c:

When processing the first '(', pos is 1. Then, when processing the ')', pos is 5. The code checks if next_paren[2] == 5. next_paren[2] is 3 (the next '('), so this is not valid. So, the code adds the ')', and pos becomes 6.

But this is incorrect, as the pair (3,5) is valid.

So, this approach is not working.

Ah, the problem is that the code is processing the string from left to right and checking each ')', but the valid pairs may be nested.

So, this approach would not capture the inner pairs.

So, this code would not work for the first sample input.

This suggests that the code needs to process the string in a way that captures nested pairs.

But how.

Another idea: the code should process the string in a way that allows for nested pairs. For example, when a valid pair is found, the code should backtrack to check for possible new pairs formed by the previous characters.

But this would require a more complex data structure.

At this point, I think the correct approach is to use a stack-based approach that tracks the indices of '(', and for each ')', check if the pair is valid. If it is, then the pair is removed, and the code continues processing from the next character after the ')'. If not, then the ') is added to the result.

But this approach would not capture nested pairs.

But in the first sample input, the pair (3,5) is valid. So, when processing the ')', the code would check if next_paren[4] == 5. If yes, then the pair is valid and is removed.

But how to ensure that the code processes the pair (3,5) before processing the pair (1,6).

This suggests that the code needs to process the string in a way that allows for nested pairs to be processed first.

But the code as written processes the string from left to right, which would process the outer pairs before the inner ones.

So, this approach would not work.

At this point, I'm stuck and need to look for a different approach.

Another idea: the valid pairs are those that can be removed, and their removal creates new valid pairs. So, the algorithm must process the pairs in the order of their nesting level, starting from the innermost.

To find the innermost pairs, we can use the next_paren array. For each '(', the earliest possible ')' that forms a valid pair is next_paren[i+1]. So, the pair (i, j) is the innermost possible for i.

So, the algorithm can be:

1. Precompute next_paren array.

2. Iterate through the string from left to right.

3. For each '(', check if next_paren[i+1] is a ')'. If yes, then this pair is valid. Remove it and proceed.

4. After removing this pair, the string is split into two parts: before i and after j. Process these parts recursively.

But this approach would be O(n) time if implemented correctly.

But how to implement this.

Let's try to model this.

We can use a list to represent the current string. We can also use a pointer to track the current position.

But for large N, this would be O(n) time, as each character is processed once.

But the code would look something like this:

current_string = list(S)

i = 0

while i < len(current_string):

   if current_string[i] == '(':

       j = next_paren[i+1]

       if j < len(current_string) and current_string[j] == ')':

           # valid pair. Remove from i to j.

           current_string = current_string[:i] + current_string[j+1:]

           # reset i to 0 to reprocess the new string.

           i = 0

       else:

           i += 1

   else:

       i += 1

But this approach is O(n^2) in the worst case, which is not feasible for large N.

So, this is not suitable.

Another idea: the problem can be solved by finding all valid pairs and then constructing the result by including the characters not in any valid pair.

But the valid pairs are those that can be removed in any order, and their removal leaves the same final string.

So, the solution is to find all pairs (i,j) where S[i] = '(', S[j] = ')', and next_paren[i+1] == j. These pairs are valid and can be removed. The remaining characters are those not in any such pair.

But how to handle overlapping pairs.

For example, in sample input 1, the valid pairs are (3,5) and (1,6) (after removal of (3,5)). But according to the next_paren array computed from the original string, (1,6) is not valid.

So, this approach would not capture the pair (1,6).

But the problem states that the final string is unique regardless of the order of removal. So, perhaps the valid pairs in the original string are sufficient to determine the final string.

But this is not the case.

At this point, I'm stuck and need to look for a different approach.

After much thought, I think the correct approach is to use a stack-based approach where we track the indices of '('. For each ')', we check if the pair is valid (using next_paren). If so, we pop the stack and do not add the pair to the result. Otherwise, we add the ')'.

But how to build the result.

The key insight is that the valid pairs are non-overlapping and can be removed in any order. So, the stack-based approach can be used to track the valid pairs and build the result.

Here's the plan:

1. Precompute the next_paren array.

2. Use a stack to track the indices of '('. For each ')', check if the stack is not empty and next_paren[i+1] == j. If yes, then this pair is valid. Pop the stack and do not add the pair to the result. Otherwise, add the ')'.

But how to track the characters between i and j.

Wait, the valid pair (i,j) includes the characters from i to j. So, when the pair is valid, all characters from i to j are removed. So, the characters between i and j should not be added to the result.

But in the stack-based approach, the code would process each character in order. So, when a valid pair is found, the code must skip adding the characters from i to j.

But how to do this.

Perhaps the stack-based approach can track the current position in the original string. When a valid pair is found, the code can skip processing the characters between i and j.

Here's how this can be implemented:

Initialize a stack. Also, have a list to build the result. Initialize a variable pos to 0.

While pos < N:

   if S[pos] is '(':

       push pos to stack.

       pos += 1.

   elif S[pos] is ')':

       if stack is not empty:

           i = stack.pop()

           j = pos

           if next_paren[i+1] == j:

               # valid pair. Skip all characters from i to j.

               pos = j + 1.

           else:

               # invalid pair. Add the ')'.

               result.append(S[pos])

               pos += 1.

       else:

           # no matching '('. Add the ')'.

           result.append(S[pos])

           pos += 1.

   else:

       # character is not a parenthesis. Add to result.

       result.append(S[pos])

       pos += 1.

This code processes the string from left to right. When a valid pair is found, it skips the characters from i to j. Otherwise, it adds the ')'.

Let's test this code with the first sample input:

Sample Input 1: a(b(d))c

Indices: 0:a, 1:(, 2:b, 3:(, 4:d, 5:), 6:), 7:c.

next_paren array:

For i=0: next_paren[0] is 1 (the first '(').

For i=1: next_paren[1] is 3 (the next '(').

For i=2: next_paren[2] is 3.

For i=3: next_paren[3] is 5 (the first ')', since S[5] is ')'.

For i=4: next_paren[4] is 5.

For i=5: next_paren[5] is 6.

For i=6: next_paren[6] is 6.

Processing:

pos starts at 0.

pos=0: character is 'a', add to result. pos=1.

pos=1: '(', push to stack. stack=[1]. pos=2.

pos=2: 'b', add to result. pos=3.

pos=3: '(', push to stack. stack=[1,3]. pos=4.

pos=4: 'd', add to result. pos=5.

pos=5: ')', stack is not empty. i=3. j=5.

Check next_paren[3+1] = next_paren[4] =5. j=5. So valid pair.

Pop stack (now stack=[1]). pos is set to 5+1=6.

pos=6: ')', stack is not empty. i=1. j=6.

Check next_paren[1+1] = next_paren[2] =3. j=6 !=3. So invalid pair.

Add ')', pos=7.

pos=7: 'c', add to result. pos=8.

The result is ['a', 'b', 'd', ')', 'c'] → "abdc c". Which is not correct.

But the sample output is "ac".

So, this approach is not working.

The problem is that the code is not handling nested pairs correctly. In the first sample, after processing the pair (3,5), the code processes pos=6, which is ')'. The stack has [1], so i=1. j=6. next_paren[2] is 3, which is not equal to j=6. So, the pair (1,6) is not considered valid, and the ') is added to the result. But in reality, after removing the pair (3,5), the pair (1,6) becomes valid.

So, this approach is not capturing the new valid pairs formed after previous removals.

Thus, the code is incorrect.

At this point, I'm stuck and need to find a different approach.

After some research, I recall that this problem is similar to the problem of removing all valid parentheses pairs, but with the condition that the pairs have no other parentheses inside. The solution to this problem can be found using a stack-based approach that tracks the indices of '(', and for each ')', checks if the pair is valid. If so, the pair is removed, and the code backtracks to check for new valid pairs formed by the previous and next characters.

But how to implement this efficiently.

Here's an alternative approach inspired by the solution to the problem of removing all valid parentheses:

We can use a stack to track the indices of '('. For each ')', if there's a matching '(', we check if the pair is valid. If yes, we remove the pair and then check the characters before and after the pair for new valid pairs.

But this requires a way to backtrack and check new pairs.

Here's the algorithm:

Initialize a stack. Also, use a list to represent the current string.

For each character in the original string:

   if it's '(', push its index onto the stack.

   elif it's ')':

       if stack is not empty:

           i = stack.pop()

           j = current index.

           if next_paren[i+1] == j:

               # valid pair. Remove from i to j.

               # update the string and check for new pairs.

               # this part is complicated.

But again, this approach requires modifying the string and is not feasible for large N.

At this point, I think the correct solution is to use a stack-based approach that tracks the indices of '(', and for each ')', checks if the pair is valid. If so, the pair is removed, and the code backtracks to check the previous character and the next character after the removed pair.

But I'm not sure how to implement this.

After much consideration, I think the correct approach is to use a stack to track the indices of '('. For each ')', if there's a matching '(', and the pair is valid, then we remove the pair and merge the left and right parts. Then, we check if the new adjacent characters form a valid pair.

But how to implement this efficiently.

Here's a possible implementation:

We can use a linked list to represent the string, allowing for efficient deletions and checks. Each node contains a character and pointers to the previous and next nodes.

But for large N, this would be feasible.

But implementing a linked list in Python for 2e5 elements is possible, but may be time-consuming.

Alternatively, we can use a list to represent the current string, and track the current position. When a valid pair is found, we remove the pair and check the new adjacent characters.

But this would require O(n^2) time in the worst case.

But given the time constraints, perhaps this is the best approach.

But given that the problem's constraints are up to 2e5, this approach would not be feasible.

At this point, I'm stuck and need to look for a different approach.

Finally, I think the correct solution is to use a stack-based approach where we track the indices of '('. For each ')', if the pair is valid, we remove it and then check the new adjacent characters. This can be done using a linked list.

Here's the code:

We'll represent the string as a doubly linked list. Each node contains a character, a prev pointer, and a next pointer.

We'll also track
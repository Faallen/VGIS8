----------------
BEHAVE data set
----------------

25 fps
resolution 640x480

10 scenarios
interactions in clip1 are labelled - markup.txt

ID1 - This is the first group which is interacting. The numbers within the square brackets [] are the members of the group.
ID2 - This is the second group which is interacting with the first (in ID1). This entry is optional.
Start - The starting frame number.
End - The ending frame number.
Label - This is the class label which describes the activity taking place (the scenarios being enacted). A short description of each scenario follows:

InGroup - The people are in a group and not moving very much
Approach - Two people or groups with one (or both) approaching the other
WalkTogether - People walking together
Meet - Two or more people meeting one another
Split - Two or more people splitting from one another
Ignore - Ignoring of one another
Chase - one group chasing another
Fight - Two or more groups fighting
RunTogether - The group is running together
Following - Being followed

Examples
ID1	ID2	Start	End	Label
[3,4]		;5826	;5926	;WalkTogether
This example is interpreted as persons with ids 3 and 4 are in a group (represented as [3,4]) which has been labelled as "WalkTogether" between frames 5826 and 5926.

ID1	ID2	Start	End	Label
[2]	[0,1]	;60296	;60349	;Approach
This example shows that person 2 is being approached by persons 0 and 1 ([0,1]) between frames 60296 and 60349. Persons 0 and 1 are in the approaching group.



Subclips
Clip 1 has been devided into 8 sequences. The AVI files are named N-M.avi and contain video frames N through M.

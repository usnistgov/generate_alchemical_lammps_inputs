# How to Contribute

We welcome contributions from external contributors, here is how
to merge code changes into this generate_alchemical_lammps_inputs.

## Getting Started

* Use your [GitLab account](https://gitlab.nist.gov/).
* [Fork](https://docs.gitlab.com/ee/user/project/repository/forking_workflow.html) this repository on GitLab.
* On your local machine,
  [clone](https://docs.gitlab.com/ee/topics/git/clone.html) your fork of
  the repository.

## Making Changes

* Make sure you [create a branch](https://docs.gitlab.com/ee/user/project/repository/branches/)
  for [your idea](http://blog.jasonmeridth.com/posts/do-not-issue-pull-requests-from-your-master-branch/),
  with the branch name relating to the feature you are going to add.
* When you are ready for others to examine and comment on your new feature,
  navigate to your fork of generate_alchemical_lammps_inputs on GitLab and open a [merge
  request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html) (MR). Note that
  after you launch a PR from one of your fork's branches, all
  subsequent commits to that branch will be added to the open pull request
  automatically.  Each commit added to the PR will be validated for
  mergability, compilation and test suite compliance; the results of these tests
  will be visible on the PR page.
* If you're providing a new feature, you must add test cases and documentation.
* When the code is ready to go, make sure you run the test suite using pytest.
* When you're ready to be considered for merging, check the "Ready to go"
  box on the PR page to let the generate_alchemical_lammps_inputs devs know that the changes are complete.
  The code will not be merged until this box is checked, the continuous
  integration returns checkmarks,
  and multiple core developers give "Approved" reviews.

# Additional Resources

* [General GitLab documentation](https://docs.gitlab.com)
* [PR best practices](http://codeinthehole.com/writing/pull-requests-and-other-good-practices-for-teams-using-github/)
* [A guide to contributing to software packages](http://www.contribution-guide.org)
* [Thinkful PR example](http://www.thinkful.com/learn/github-pull-request-tutorial/#Time-to-Submit-Your-First-PR)

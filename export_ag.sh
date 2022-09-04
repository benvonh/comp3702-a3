if test -f "ag.zip"; then
  rm ag.zip
fi
rm control/environment.py control/state.py
cp environment.py control/environment.py
cp state.py control/state.py
zip ag.zip testcases/* control/* autograder.py constants.py environment.py state.py run_autograder setup.sh tester.py
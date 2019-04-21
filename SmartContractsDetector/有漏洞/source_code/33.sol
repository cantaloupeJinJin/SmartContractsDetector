pragma solidity ^0.4.25;

contract DontWantYourEtherForFree {
    function someUsefullFunction() public payable {
        // do some meaningful work
    }

    function () {
        // dont receive ether via fallback
    }
}

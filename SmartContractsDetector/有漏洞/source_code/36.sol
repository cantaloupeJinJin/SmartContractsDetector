pragma solidity ^0.4.25;

contract BreakThisHash {
    bytes32 hash;
    uint birthday;
    constructor(bytes32 _hash) public payable {
        hash = _hash;
        birthday = now;
    }

    function kill(bytes password) external {
        if (sha3(password) != hash) {
            throw;
        }
        suicide(msg.sender);
    }

    function hashAge() public constant returns(uint) {
        return(now - birthday);
    }
}
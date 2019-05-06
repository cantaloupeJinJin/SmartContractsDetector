// solhint-disable
//pragma solidity ^0.4.4;


contract A {
    address public owner;
    uint public last_completed_migration;
  
    modifier restricted() {
        if (msg.sender == owner) _;
    }
  
    function A() public {
        owner = msg.sender;
    }
  
    function setCompleted(uint completed) public restricted {
        last_completed_migration = completed;
    }
  
    function upgrade(address new_address) public restricted {
        A upgraded = A(new_address);
        upgraded.setCompleted(last_completed_migration);
    }
}

/**
 * Source Code first verified at https://etherscan.io on Sunday, May 5, 2019
 (UTC) */

//pragma solidity 0.4.19;

contract Ownable {
    address public owner;

  function Ownable() public {
    owner = msg.sender;
  }

  modifier onlyOwner() {
    require(msg.sender == owner);
    _;
  }
}

contract ContractReceiver {
    function tokenFallback(address _from, uint _value) public pure returns(address) {
       if (_value != 0) return _from;
    }
}


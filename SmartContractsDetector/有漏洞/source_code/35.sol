pragma solidity ^0.4.25;

contract GasLimitAndLoops {
    address[] internal investors;
    mapping(address => uint) public balances;
    
    function collectInvestment() external returns (uint) {
        uint sum = 0;
        for (uint i = 0; i < investors.length; i++) {
            sum += balances[investors[i]];
            balances[investors[i]] = 0;
        }
        msg.sender.transfer(sum);
    }
    
    function invest() external payable {
        investors.push(msg.sender);
        balances[msg.sender] = msg.value;
    }
}
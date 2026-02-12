from ecommerce_audit.reward.reward_functions import (
    attribute_consistency_reward,
    compliance_reward,
    format_reward,
)


GOOD = '{"category":"shoes","attributes":{"color":"black"},"violation":false,"reason":"图文一致","violation_types":[]}'
BAD_FORMAT = '{"category":"shoes"'
BAD_ATTR = '{"category":"shoes","attributes":{"color":"red"},"violation":false,"reason":"图文一致","violation_types":[]}'


def test_format_reward():
    assert format_reward(GOOD) > format_reward(BAD_FORMAT)


def test_attribute_reward():
    assert attribute_consistency_reward(GOOD, GOOD) > attribute_consistency_reward(BAD_ATTR, GOOD)


def test_compliance_reward():
    assert compliance_reward(GOOD, GOOD) > -1

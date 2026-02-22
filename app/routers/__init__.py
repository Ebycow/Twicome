from . import comments, misc, quiz, search, stats

ALL_ROUTERS = [
    comments.router,
    search.router,
    stats.router,
    quiz.router,
    misc.router,
]

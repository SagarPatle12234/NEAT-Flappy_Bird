import pygame
import neat
import os
import pickle
from main import Bird, Pipe, Base, WIN_WIDTH, WIN_HEIGHT, FLOOR, BG_IMG, draw_window

pygame.font.init()

def eval_winner(net, config):
    bird = Bird(230, 350)
    base = Base(FLOOR)
    pipes = [Pipe(600)]
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()
    score = 0

    run = True
    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        pipe_ind = 0
        if len(pipes) > 1 and bird.x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
            pipe_ind = 1

        # Use the loaded network to decide whether to jump
        output = net.activate((
            bird.y,
            abs(bird.y - pipes[pipe_ind].height),
            abs(bird.y - pipes[pipe_ind].bottom)
        ))

        if output[0] > 0.5:
            bird.jump()

        bird.move()
        base.move()

        rem = []
        add_pipe = False
        for pipe in pipes:
            pipe.move()
            if pipe.collide(bird):
                run = False

            if not pipe.passed and pipe.x < bird.x:
                pipe.passed = True
                add_pipe = True

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

        if add_pipe:
            score += 1
            pipes.append(Pipe(600))

        for r in rem:
            pipes.remove(r)

        if bird.y + bird.img.get_height() >= FLOOR or bird.y < 0:
            run = False

        draw_window(win, bird, pipes, base, score)

def load_winner_model(config_path, model_path):
    # Load the config
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    # Load the saved best genome
    with open(model_path, "rb") as f:
        genome = pickle.load(f)

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    eval_winner(net, config)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, "Neat_Config.txt")
    model_file = os.path.join(local_dir, "best_flappy_bird.pkl")

    load_winner_model(config_file, model_file)
